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

plt.rcParams["figure.dpi"] = 120
torch.set_float32_matmul_precision("high")

# -----------------------------
# User-editable configuration (cleaned / deduplicated)

# -----------------------------
DATA_DIR = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/")
PT_PATH = DATA_DIR / "preprocessed_train_3000hvg_train.pt"
GENE_PATH = DATA_DIR / "sc_processed_genes_train_val_set.h5ad"
ISOFORM_PATH = DATA_DIR / "sc_processed_transcripts_train_val_set.h5ad"
OUT_DIR = Path("/zhome/af/a/221977/Blackhole/Results/MLP_PCA_3/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
compute_pca = True
n_comps = 50

# dataset / training
MAX_SAMPLES = None
MIN_COUNTS = 20
MAX_GENES = None
MAX_ISOFORMS = None
VAL_FRACTION = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 50

# optimizer / model
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIMS = [1024, 524, 256, 128]
USE_PREPROCESSED = True
DROPOUT = 0.5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENE_ID_COL = "gene_name"
ISOFORM_ID_COL = "transcript_id"
PROPORTION_BATCH = 64
LOAD_CHUNK_SIZE = 64
COUNT_CHUNK_SIZE = 128

# Set the PCA flag = True to train on PCA embeddings
# Set the number of components considered for PCA


# -----------------------------
# Main pipeline
# -----------------------------

def main_mlp():
    set_seed(SEED)

    print("Loading preprocessed data...")
    genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids, gene_ids = load_standard_hvg(PT_PATH, compute_pca=compute_pca)

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

    model = MLP(
        input_dim=genes_tensor.shape[1],
        hidden_dims=HIDDEN_DIMS,
        out_dim=isoform_proportions.shape[1],
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
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

    # per-run output folder
    mlp_out = OUT_DIR / "MLP_GeneAware_PCA"
    mlp_out.mkdir(parents=True, exist_ok=True)


    checkpoint_path = mlp_out / "standardMLP_PCA_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dims": HIDDEN_DIMS, "max_samples": MAX_SAMPLES},
            "history": history,
            # "isoform_ids": isoform_ids,
        },
        checkpoint_path,
    )
    print(f"Saved MLP checkpoint to {checkpoint_path}")
    plot_history(history, mlp_out / "MLP_GeneAware_PCA_Training.png")
    # plot_residuals(preds, targets, mlp_out / "standardMLP_PCA_Evaluation.png")
    plot_residuals(history['val_preds'], history['val_targets'], mlp_out / "MLP_GeneAware_PCA_Evaluation.png")

if __name__ == "__main__":
    main_mlp()