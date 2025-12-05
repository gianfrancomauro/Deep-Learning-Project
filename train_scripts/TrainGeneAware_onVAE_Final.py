"""
Train a GeneAware MLP on precomputed VAE embeddings to predict isoform proportions.
Based on TrainMLPGA_3kHVG.py structure.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow

from scripts.models import GeneAwareMLP
from scripts.train import train_model
from scripts.utils import classification_error, pearson_corr, set_seed, EarlyStopping
from scripts.plots import plot_history, plot_residuals
from scripts.data import load_standard_hvg, load_embeddings

torch.set_float32_matmul_precision("high")

# -----------------------------
# Configuration
# -----------------------------

# Paths
TRAIN_DATA_PATH = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/preprocessed_train_3000hvg_train.pt")
VAE_TRAIN_EMB_PATH = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/vae_latent_embeddings_preprocessed.pt")
OUTPUT_DIR = Path("Results/MLP_VAE")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
HIDDEN_DIMS = [1024, 512, 256, 128]
DROPOUT = 0.4
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
STANDARDIZE_EMB = True
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MLflow setup
mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
mlflow.set_experiment("MLP_VAE_Training")
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

def main():
    set_seed(SEED)

    print("=" * 80)
    print("Training GeneAware MLP on VAE Embeddings")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden dims: {HIDDEN_DIMS}")
    print()

    # Load preprocessed data (isoform targets and metadata)
    print("Loading preprocessed training data...")
    genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids, gene_ids = load_standard_hvg(
        pt_path=TRAIN_DATA_PATH,
        max_samples=None,
        seed=SEED,
    )
    print(f"  Gene tensor shape: {genes_tensor.shape}")
    print(f"  Isoform proportions shape: {isoform_proportions.shape}")

    # Load VAE embeddings (replace genes_tensor as input)
    print("\nLoading VAE embeddings...")
    Z, emb_stats = load_embeddings(VAE_TRAIN_EMB_PATH, standardize=STANDARDIZE_EMB)
    print(f"  VAE embeddings shape: {Z.shape}")
    print(f"  Standardized: {STANDARDIZE_EMB}")

    # Verify alignment
    if Z.shape[0] != isoform_proportions.shape[0]:
        raise ValueError(
            f"Embeddings ({Z.shape[0]}) and isoform targets ({isoform_proportions.shape[0]}) "
            "have mismatched sample counts. Ensure VAE embeddings were generated from the same data."
        )

    # Split into train/val
    print(f"\nSplitting data (train/val: {1-VAL_FRACTION:.0%}/{VAL_FRACTION:.0%})...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Z))
    val_size = max(1, int(len(Z) * VAL_FRACTION))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_dataset = TensorDataset(Z[train_idx], isoform_proportions[train_idx])
    val_dataset = TensorDataset(Z[val_idx], isoform_proportions[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Initialize model
    model = GeneAwareMLP(
        input_dim=Z.shape[1],
        hidden_dims=HIDDEN_DIMS,
        isoform_dim=isoform_proportions.shape[1],
        gene_index_per_iso=torch.from_numpy(iso_to_gene_index),  # Pass tensor
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
    loss_fn = nn.MSELoss()

    print("\nStarting training...")
    print("=" * 80)

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "hidden_dims": str(HIDDEN_DIMS),
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "val_fraction": VAL_FRACTION,
            "standardize_emb": STANDARDIZE_EMB,
            "seed": SEED,
            "device": DEVICE,
            "input_dim": Z.shape[1],
            "output_dim": isoform_proportions.shape[1],
        })

        # Train model
        history = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            NUM_EPOCHS,
            scheduler=scheduler,
            early_stopper=stopper,
            device=DEVICE,
        )

        # Plot training history
        plot_history(history, OUTPUT_DIR / "MLP_VAE_Training.png")
        mlflow.log_artifact(OUTPUT_DIR / "MLP_VAE_Training.png")

        # Final evaluation on validation set
        print("\n" + "=" * 80)
        print("Final Evaluation on Validation Set")
        print("=" * 80)
        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE).float()
                targets = targets.to(DEVICE).float()
                outputs = model(inputs)
                preds_list.append(outputs.cpu())
                targets_list.append(targets.cpu())

        preds = torch.cat(preds_list).flatten()
        targets = torch.cat(targets_list).flatten()

        val_rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
        val_corr = pearson_corr(preds, targets)

        # Classification error (requires tensor version of iso_to_gene_index)
        iso_to_gene_index_tensor = torch.from_numpy(iso_to_gene_index).long()
        preds_full = torch.cat(preds_list)  # Keep shape for classification error
        targets_full = torch.cat(targets_list)
        val_class_err = classification_error(preds_full, targets_full, iso_to_gene_index_tensor)

        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation Pearson: {val_corr:.4f}")
        print(f"Validation Classification Error: {val_class_err:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            "final_val_rmse": val_rmse.item(),
            "final_val_pearson": val_corr.item(),
            "final_val_class_error": val_class_err,
        })

        # Plot residuals
        plot_residuals(preds, targets, OUTPUT_DIR / "MLP_VAE_Residuals.png")
        mlflow.log_artifact(OUTPUT_DIR / "MLP_VAE_Residuals.png")

        # Save checkpoint
        checkpoint_path = OUTPUT_DIR / "MLP_VAE_weights.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_dims": HIDDEN_DIMS,
                    "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE,
                    "epochs": NUM_EPOCHS,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "val_fraction": VAL_FRACTION,
                    "standardize_emb": STANDARDIZE_EMB,
                    "seed": SEED,
                    "device": DEVICE,
                    "input_dim": Z.shape[1],
                    "output_dim": isoform_proportions.shape[1],
                },
                "history": history,
                "gene_to_iso_map": gene_to_iso_map,
                "isoform_ids": isoform_ids,
                "iso_to_gene_index": iso_to_gene_index,  # numpy array
                "embedding_stats": {k: v.cpu() for k, v in emb_stats.items()} if emb_stats else {},
            },
            checkpoint_path,
        )
        print(f"\nSaved checkpoint to {checkpoint_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
