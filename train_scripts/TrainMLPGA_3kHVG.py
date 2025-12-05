from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from scripts.data import load_standard_hvg
from scripts.models import MLP, GeneAwareMLP
from scripts.train import train_model
from scripts.utils import set_seed, pearson_corr, EarlyStopping
from scripts.plots import plot_history, plot_residuals

plt.rcParams["figure.dpi"] = 60
torch.set_float32_matmul_precision("high")

# add script-style imports like TrainGeneAware_Final
from scripts.utils import set_seed

# -----------------------------
# User-editable configuration (cleaned / deduplicated)
# -----------------------------

# Set to True to use preprocessed .pt file (faster), False to load from raw .h5ad files
DATA_DIR = Path("/dtu/blackhole/19/221977/train_val_split_data/")
PT_PATH = DATA_DIR / "preprocessed_train_3000hvg_train.pt"

# Preprocessed data path (if USE_PREPROCESSED=True)

# Raw data paths (if USE_PREPROCESSED=False)
GENE_PATH = DATA_DIR / "sc_processed_genes_train_val_set.h5ad"
ISOFORM_PATH = DATA_DIR / "sc_processed_transcripts_train_val_set.h5ad"

compute_pca = False

# toy data paths for quick testing
# GENE_PATH = Path("/zhome/af/a/221977/Blackhole/scGPT_embeddings/toy.h5ad")
# ISOFORM_PATH = Path("/zhome/af/a/221977/Blackhole/scGPT_embeddings/toy.h5ad")

OUT_DIR = Path("/dtu/blackhole/19/221977/Results/MLP_vs_GeneAware_MLP")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# dataset / training
MAX_SAMPLES = None
MIN_COUNTS = 20 # min counts per isoform
MAX_GENES = None
MAX_ISOFORMS = None
VAL_FRACTION = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 100

# optimizer / model
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIMS = [512, 256, 128, 128]
DROPOUT = 0.25
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Main pipeline
# -----------------------------


def main_geneaware():
    set_seed(SEED)

    # Load data either from preprocessed .pt file or raw .h5ad files
    print("Loading preprocessed data...")
    genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids, gene_ids = load_standard_hvg(
        PT_PATH
    )

    print(f"Gene tensor shape: {genes_tensor.shape}")
    print(f"Isoform tensor shape: {isoform_proportions.shape}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(genes_tensor))
    val_size = max(1, int(len(genes_tensor) * VAL_FRACTION))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_dataset = TensorDataset(genes_tensor[train_idx], isoform_proportions[train_idx])
    val_dataset = TensorDataset(genes_tensor[val_idx], isoform_proportions[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = GeneAwareMLP(
        input_dim=genes_tensor.shape[1],
        hidden_dims=HIDDEN_DIMS,
        isoform_dim=isoform_proportions.shape[1],
        gene_index_per_iso=torch.from_numpy(iso_to_gene_index),
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=20, min_delta=1e-4, mode="min")
    loss_fn = nn.MSELoss()

    params = {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "num_epochs": NUM_EPOCHS,
        "scheduler": scheduler,
        "early_stopper": stopper,
        "device": DEVICE,
    }

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

    # ensure the gene-aware outputs go to a separate folder
    ga_out = OUT_DIR / "GeneAware"
    ga_out.mkdir(parents=True, exist_ok=True)
    plot_history(history, ga_out / "GeneAware_Training.png")


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
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
    corr = pearson_corr(preds, targets)
    print(f"GeneAware Validation RMSE: {rmse:.4f}")
    print(f"GeneAware Validation Pearson: {corr:.4f}")


    plot_residuals(preds, targets, ga_out / "GeneAware_Evaluation.png")

    checkpoint_path = ga_out / "GeneAware_Full_MLP_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dims": HIDDEN_DIMS,
                "min_counts": MIN_COUNTS,
                "max_genes": MAX_GENES,
                "max_isoforms": MAX_ISOFORMS,
                "max_samples": MAX_SAMPLES,
            },
            "history": history,
            "sorted_gene_ids": list(gene_to_iso_map.keys()),
            "gene_to_iso_map": gene_to_iso_map,
            "isoform_ids": isoform_ids,
        },
        checkpoint_path,
    )
    print(f"Saved GeneAware checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main_geneaware()