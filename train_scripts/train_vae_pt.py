"""
Train VAE using preprocessed .pt files with MSE loss (for normalized data).
This uses Gaussian VAE loss instead of Negative Binomial loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from scripts.models import VAE
from scripts.data import GenesDataset, load_standard_hvg
from scripts.utils import EarlyStopping, get_device, set_seed

# === Configuration ===
TRAIN_VAL_PT_PATH = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/preprocessed_train_3000hvg_train.pt")
OUT_DIR = Path("/zhome/af/a/221977/Blackhole/train_val_split_data")

# VAE Architecture
LATENT_DIM = 3000
HIDDEN_DIMS = [1024, 512, 256, 128]

# Training Parameters
VAL_FRACTION = 0.1
SEED = 42
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 100
WARMUP_EPOCHS = 50
BETA = 0.001  # Lower beta for normalized data (KL term is sensitive)

torch.set_float32_matmul_precision("high")


def vae_gaussian_loss(x, recon_x, mu, logvar, beta=1.0):
    """
    Gaussian VAE loss for normalized/continuous data.

    Args:
        x: Input data [batch, features]
        recon_x: Reconstructed data [batch, features] (px_mu from VAE)
        mu: Latent mean [batch, latent_dim]
        logvar: Latent log-variance [batch, latent_dim]
        beta: Weight for KL divergence term

    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # MSE reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def build_dataloaders_from_pt(pt_path: Path, val_fraction: float, batch_size: int, seed: int):
    """Load preprocessed .pt file and split into train/val for VAE training."""
    print(f"[build_dataloader] Loading preprocessed data from: {pt_path}")

    genes_tensor, isoform_proportions, _, _, _, _ = load_standard_hvg(
        pt_path=pt_path,
        max_samples=None,
        seed=seed,
    )

    n_samples = genes_tensor.shape[0]
    n_genes = genes_tensor.shape[1]
    n_isoforms = isoform_proportions.shape[1]

    print(f"[build_dataloader] Loaded: {n_samples} samples, {n_genes} genes, {n_isoforms} isoforms")
    print(f"[build_dataloader] Gene data stats: min={genes_tensor.min():.4f}, max={genes_tensor.max():.4f}, mean={genes_tensor.mean():.4f}")

    # Split into train/val
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    val_count = max(1, int(n_samples * val_fraction))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    print(f"[build_dataloader] Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    # Create datasets - VAE uses genes as input
    train_dataset = GenesDataset(genes_tensor[train_idx], isoform_proportions[train_idx])
    val_dataset = GenesDataset(genes_tensor[val_idx], isoform_proportions[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, n_genes


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    beta_max: float = 1.0,
    warmup_epochs: int = 3,
    early_stopper = None,
):
    """Train VAE with Gaussian (MSE) loss for normalized data."""
    print("\n" + "=" * 80)
    print("Starting VAE training with MSE loss (for normalized data)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Beta max: {beta_max}")
    print(f"Warmup epochs: {warmup_epochs}")
    print()

    model.to(device)
    printed_debug_shapes = False

    history = {
        "loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
        "beta": [],
    }

    for epoch in range(1, num_epochs + 1):
        # KL warm-up
        beta = beta_max * min(1.0, epoch / float(warmup_epochs))

        # ========== TRAINING ==========
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_batch, _ = batch
            else:
                x_batch = batch

            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            px_mu, px_r, mu, logvar = model(x_batch)

            if not printed_debug_shapes:
                print("[DEBUG] Tensor shapes:")
                print(f"  x_batch: {x_batch.shape}")
                print(f"  px_mu (recon): {px_mu.shape}")
                print(f"  mu (latent): {mu.shape}")
                print(f"  logvar: {logvar.shape}")
                print(f"  x_batch range: [{x_batch.min():.4f}, {x_batch.max():.4f}]")
                printed_debug_shapes = True

            # Gaussian VAE loss (MSE reconstruction)
            loss, recon_loss, kl_loss = vae_gaussian_loss(
                x_batch, px_mu, mu, logvar, beta=beta
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_train_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[Train | Epoch {epoch} | Batch {batch_idx+1}] "
                    f"loss: {loss.item():.4f} | "
                    f"recon: {recon_loss.item():.4f} | "
                    f"kl: {kl_loss.item():.4f}"
                )

        avg_loss = epoch_loss / num_train_batches
        avg_recon = epoch_recon / num_train_batches
        avg_kl = epoch_kl / num_train_batches

        # ========== VALIDATION ==========
        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x_batch, _ = batch
                else:
                    x_batch = batch

                x_batch = x_batch.to(device)
                px_mu, px_r, mu, logvar = model(x_batch)

                val_loss, val_recon, val_kl = vae_gaussian_loss(
                    x_batch, px_mu, mu, logvar, beta=beta
                )

                val_loss_sum += val_loss.item()
                val_recon_sum += val_recon.item()
                val_kl_sum += val_kl.item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / num_val_batches
        avg_val_recon = val_recon_sum / num_val_batches
        avg_val_kl = val_kl_sum / num_val_batches

        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["val_loss"].append(avg_val_loss)
        history["val_recon_loss"].append(avg_val_recon)
        history["val_kl_loss"].append(avg_val_kl)
        history["beta"].append(beta)

        print(
            f"\n[Epoch {epoch}/{num_epochs}] "
            f"beta={beta:.4f} | "
            f"train_loss: {avg_loss:.4f} (recon: {avg_recon:.4f}, kl: {avg_kl:.4f}) | "
            f"val_loss: {avg_val_loss:.4f} (recon: {avg_val_recon:.4f}, kl: {avg_val_kl:.4f})"
        )

        if early_stopper and early_stopper(avg_val_loss):
            print(f"\n[EarlyStopping] Stopped at epoch {epoch}")
            break

    print("\n" + "=" * 80)
    print("VAE Training Completed")
    print("=" * 80)

    return history


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}\n")

    # Load data
    train_loader, val_loader, input_dim = build_dataloaders_from_pt(
        pt_path=TRAIN_VAL_PT_PATH,
        val_fraction=VAL_FRACTION,
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    print(f"Input dimension (num genes): {input_dim}\n")

    # Initialize VAE
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS)
    print("VAE architecture:")
    print(model)
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    early_stopper = EarlyStopping(patience=10, min_delta=0.0, mode="min")

    # Train VAE
    history = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        beta_max=BETA,
        warmup_epochs=WARMUP_EPOCHS,
        early_stopper=early_stopper,
    )

    # Save model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "vae_trained_on_preprocessed.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel weights saved to: {ckpt_path}")


if __name__ == "__main__":
    main()