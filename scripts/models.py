"""
Model definitions.
"""
from typing import List

import math
import copy
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.25):
        super().__init__()
        modules = []
        dims = [input_dim] + hidden_dims
        for in_size, out_size in zip(dims[:-1], dims[1:]):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=dropout))
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GeneAwareMLP(nn.Module):
    """
    MLP that normalizes outputs within each gene across its isoforms.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], isoform_dim: int, gene_index_per_iso: torch.Tensor, dropout: float = 0.25):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], isoform_dim))
        self.network = nn.Sequential(*layers)
        self.register_buffer("iso_to_gene_index", gene_index_per_iso.long())
        self.n_genes = int(self.iso_to_gene_index.max().item() + 1)

    def _gene_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        idx = self.iso_to_gene_index.unsqueeze(0).expand(logits.size(0), -1)
        exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        denom = torch.zeros(logits.size(0), self.n_genes, device=logits.device, dtype=logits.dtype)
        denom.scatter_add_(1, idx, exp_logits)
        norm = torch.gather(denom, 1, idx).clamp_min_(1e-8)
        return exp_logits / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return self._gene_softmax(logits)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # Handle odd d_model dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Scale input by sqrt(d_model) as per original paper
        x = x * math.sqrt(x.size(-1))
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    dropout_p = 0.0
    if dropout is not None:
        if isinstance(dropout, nn.Dropout):
            dropout_p = dropout.p
        elif isinstance(dropout, float):
            dropout_p = dropout

    if mask is not None and mask.dtype != torch.bool:
        mask = mask.bool()

    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout_p)
    return output, None


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class TransformerEncoder(nn.Module):
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerIsoformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        isoform_dim: int,
        gene_index_per_iso: torch.Tensor,
        d_model: int = 256,
        n_head: int = 4,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.register_buffer("iso_to_gene_index", gene_index_per_iso.long())
        self.n_genes = int(self.iso_to_gene_index.max().item() + 1)

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learned gene embeddings to give each token an identity (no positional encoding)
        self.gene_embed = nn.Embedding(self.n_genes, d_model)
        self.register_buffer("gene_token_index", torch.arange(self.n_genes))
        
        # Transformer Encoder
        attn = MultiHeadedAttention(n_head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = TransformerEncoder(EncoderLayer(d_model, attn, ff, dropout), n_layers)
        
        # Final projection to isoform dimension
        # We take the mean of the sequence output as the representation
        # Or we could use a [CLS] token. For simplicity, mean pooling.
        self.output_proj = nn.Linear(d_model, isoform_dim)

    def _gene_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        idx = self.iso_to_gene_index.unsqueeze(0).expand(logits.size(0), -1)
        x = logits - logits.max(dim=1, keepdim=True).values
        exp_x = torch.exp(x)
        denom = torch.zeros(logits.size(0), self.n_genes, device=logits.device)
        denom.scatter_add_(1, idx, exp_x)
        norm = torch.gather(denom, 1, idx).clamp_min(1e-8)
        return exp_x / norm

    def forward(self, x: torch.Tensor, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        seq_len = x.size(1)
        x = self.input_proj(x)

        gene_idx = self.gene_token_index[:seq_len]
        x = x + self.gene_embed(gene_idx).unsqueeze(0)

        x = self.encoder(x, mask)
        x = x.mean(dim=1)

        logits = self.output_proj(x)
        return self._gene_softmax(logits)


class LSTMIsoformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        isoform_dim: int,
        gene_index_per_iso: torch.Tensor,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Linear(lstm_output_dim, isoform_dim),
        )
        self.register_buffer("iso_to_gene_index", gene_index_per_iso.long())
        self.n_genes = int(self.iso_to_gene_index.max().item() + 1)

    def _gene_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        idx = self.iso_to_gene_index.unsqueeze(0).expand(logits.size(0), -1)
        x = logits - logits.max(dim=1, keepdim=True).values
        exp_x = torch.exp(x)
        denom = torch.zeros(logits.size(0), self.n_genes, device=logits.device)
        denom.scatter_add_(1, idx, exp_x)
        norm = torch.gather(denom, 1, idx).clamp_min(1e-8)
        return exp_x / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  
        _, (h_last, _) = self.lstm(x)
        if self.bidirectional:
            h_last = torch.cat([h_last[-2], h_last[-1]], dim=-1)
        else:
            h_last = h_last[-1]
        logits = self.fc(h_last)
        return self._gene_softmax(logits)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=512, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [4096, 2048, 1024]

        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # NB parameters: px_scale (mean) and px_r (dispersion)
        self.px_scale = nn.Linear(in_dim, input_dim)
        self.px_r = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.fc_logvar.weight)
        if self.fc_logvar.bias is not None:
            nn.init.zeros_(self.fc_logvar.bias)

        nn.init.normal_(self.px_scale.weight, mean=0.0, std=1e-3)
        if self.px_scale.bias is not None:
            nn.init.zeros_(self.px_scale.bias)

        nn.init.zeros_(self.px_r)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        NB-style decoder: from z produce the parameters of the Negative Binomial.

        Returns:
        - px_mu: NB mean, shape [batch_size, input_dim], strictly positive
        - px_r: NB dispersion per gene, shape [input_dim], positive
        """
        h = self.decoder(z)

        # NB mean: softplus to positivity
        px_mu = F.softplus(self.px_scale(h))

        # NB dispersion 
        px_r = torch.exp(self.px_r)

        return px_mu, px_r

    def forward(self, x):
        """
        Full VAE forward pass:
        x → (mu, logvar) → z → (px_mu, px_r)

        Returns:
        - px_mu, px_r: NB parameters for p(x | z)
        - mu, logvar: parameters of q(z | x) used for KL
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        px_mu, px_r = self.decode(z)
        return px_mu, px_r, mu, logvar

    def encode_only(self, x, use_mu=False):
        """
        Utility used in encode_vae.py:
        returns only the latent code z or mu.
        """
        mu, logvar = self.encode(x)
        if use_mu:
            return mu
        else:
            return self.reparameterize(mu, logvar)


# ------- Loss functions -------

def nb_log_likelihood(x, mu, r, eps=1e-8):
    """
    Element-wise Negative Binomial log-likelihood.

    x  : tensor [batch_size, n_genes]
    mu : NB mean, same shape as x
    r  : NB dispersion, shape [n_genes]
    """
    mu = mu.clamp(min=eps)
    r = r.clamp(min=eps)

    # Standard NB formula parametrized by (mu, r)
    t1 = torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
    t2 = r * (torch.log(r + eps) - torch.log(mu + r + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(mu + r + eps))
    return t1 + t2 + t3  # log p(x | mu, r)


def vae_nb_loss(x, px_mu, px_r, mu, logvar, beta=1.0):
    """
    ELBO-style loss with:
    - Negative Binomial reconstruction
    - KL divergence for diagonal Gaussian vs N(0, I)
    - beta weight on KL term
    """
    # Reconstruction term: negative expected log-likelihood
    recon_log_prob = nb_log_likelihood(x, mu=px_mu, r=px_r)

    # Sum over genes, average over samples
    recon_loss = -recon_log_prob.sum(dim=1).mean()

    # KL divergence for diagonal q(z|x)
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )
    kl_loss = kl_per_sample.mean()

    loss_tot = recon_loss + beta * kl_loss
    return loss_tot, recon_loss, kl_loss


# Optional: keep the old MSE-based loss for compatibility,
# even though the NB-based training uses vae_nb_loss.
def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    Old version: MSE reconstruction + beta * KL.
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )
    kl_loss = torch.mean(kl_per_sample)

    loss_tot = recon_loss + beta * kl_loss
    return loss_tot, recon_loss, kl_loss
        