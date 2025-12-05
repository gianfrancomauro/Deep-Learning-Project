"""
Encode preprocessed .pt file data using trained VAE to generate latent embeddings.
This ensures embeddings match the exact same data used for MLP training.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from scripts.models import VAE
from scripts.data import GenesDataset, load_standard_hvg
from scripts.utils import get_device, set_seed

# === Configuration ===
# Input: preprocessed .pt file
INPUT_PT_PATH = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/preprocessed_train_3000hvg_test.pt")

# VAE checkpoint (trained on the same data)
VAE_CKPT_PATH = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/vae_trained_on_preprocessed.pt")

# Output: latent embeddings
EMB_OUT_DIR = Path("/zhome/af/a/221977/Blackhole/train_val_split_data")
EMB_OUT_NAME = "vae_latent_embeddings_preprocessed_test.pt"

# VAE Architecture (must match training)
LATENT_DIM = 512
HIDDEN_DIMS = [1024, 512, 256, 128]

# Encoding parameters
BATCH_SIZE = 32
SEED = 42
USE_MU = True  # Use mean of latent distribution (deterministic encoding)

torch.set_float32_matmul_precision("high")


def build_dataloader_for_encoding(pt_path: Path, batch_size: int, seed: int):
    """
    Load ALL samples from preprocessed .pt file for encoding.
    IMPORTANT: shuffle=False to maintain order for downstream MLP training.

    Returns:
        dataloader, input_dim (number of genes), n_samples
    """
    print(f"[encode] Loading preprocessed data from: {pt_path}")

    # Load using load_standard_hvg
    genes_tensor, isoform_proportions, _, _, _, _ = load_standard_hvg(
        pt_path=pt_path,
        max_samples=None,
        seed=seed,
    )

    n_samples = genes_tensor.shape[0]
    n_genes = genes_tensor.shape[1]
    n_isoforms = isoform_proportions.shape[1]

    print(f"[encode] Loaded: {n_samples} samples, {n_genes} genes, {n_isoforms} isoforms")
    print(f"[encode] Will encode ALL {n_samples} samples (no subsetting)")

    # Create dataset with ALL samples
    dataset = GenesDataset(genes_tensor, isoform_proportions)

    # CRITICAL: shuffle=False to keep original order!
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: maintain order
        drop_last=False,
    )

    return loader, n_genes, n_samples


def load_trained_vae(input_dim: int, ckpt_path: Path, device: torch.device):
    """
    Create VAE with same hyperparameters as training and load weights.
    """
    print(f"[encode] Loading VAE from checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")

    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[encode] VAE loaded successfully")
    return model


def encode_dataset(model: VAE, dataloader: DataLoader, device: torch.device, use_mu: bool = True):
    """
    Pass all samples through VAE encoder to get latent embeddings.

    Args:
        use_mu: If True, use mean of latent distribution (deterministic).
                If False, sample from distribution (stochastic).

    Returns:
        Tensor of shape [n_samples, latent_dim]
    """
    print(f"[encode] Encoding dataset (use_mu={use_mu})...")

    all_z = []
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract genes from batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_batch, _ = batch
            else:
                x_batch = batch

            x_batch = x_batch.to(device)

            # Encode using VAE
            z_batch = model.encode_only(x_batch, use_mu=use_mu)

            # Debug first batch
            if batch_idx == 0:
                print("[encode] First batch shapes:")
                print(f"  x_batch (genes): {x_batch.shape}")
                print(f"  z_batch (latent): {z_batch.shape}")

            all_z.append(z_batch.cpu())

            # Progress
            if (batch_idx + 1) % 100 == 0:
                print(f"[encode] Processed {(batch_idx + 1) * dataloader.batch_size} samples...")

    # Concatenate all batches
    Z_latent = torch.cat(all_z, dim=0)
    print(f"[encode] Final latent embeddings shape: {Z_latent.shape}")

    return Z_latent


def main():
    set_seed(SEED)
    device = get_device()
    print(f"[encode] Device: {device}")
    print(f"[encode] Input .pt file: {INPUT_PT_PATH}")
    print(f"[encode] VAE checkpoint: {VAE_CKPT_PATH}")
    print()

    # Load data
    loader, input_dim, n_samples = build_dataloader_for_encoding(
        pt_path=INPUT_PT_PATH,
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    print(f"[encode] Input dimension: {input_dim}")
    print(f"[encode] Total samples: {n_samples}\n")

    # Load trained VAE
    model = load_trained_vae(
        input_dim=input_dim,
        ckpt_path=VAE_CKPT_PATH,
        device=device
    )
    print()

    # Encode dataset
    Z_latent = encode_dataset(model, loader, device, use_mu=USE_MU)

    # Statistics
    print()
    print("=" * 80)
    print("Latent Embedding Statistics:")
    print("=" * 80)
    print(f"Shape: {Z_latent.shape}")
    print(f"Mean (per dimension, first 5): {Z_latent.mean(dim=0)[:5].tolist()}")
    print(f"Std (per dimension, first 5): {Z_latent.std(dim=0)[:5].tolist()}")
    print(f"Overall mean: {Z_latent.mean():.4f}")
    print(f"Overall std: {Z_latent.std():.4f}")
    print()

    # Save embeddings
    EMB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = EMB_OUT_DIR / EMB_OUT_NAME
    torch.save(Z_latent, emb_path)
    print(f"[encode] Embeddings saved to: {emb_path}")

    # Also save as .npy for convenience
    npy_path = str(emb_path).replace(".pt", ".npy")
    np.save(npy_path, Z_latent.numpy())
    print(f"[encode] Embeddings also saved to: {npy_path}")

    print()
    print("=" * 80)
    print("Encoding Complete!")
    print("=" * 80)
    print(f"You can now use {emb_path} for MLP training")


if __name__ == "__main__":
    main()