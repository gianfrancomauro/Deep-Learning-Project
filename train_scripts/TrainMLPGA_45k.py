from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from scripts.data import load_geneaware
from scripts.models import MLP
from scripts.train import train_model
from scripts.utils import set_seed, EarlyStopping
from scripts.plots import plot_history, plot_residuals

plt.rcParams["figure.dpi"] = 120
torch.set_float32_matmul_precision("high")

DATA_DIR = Path("/dtu/blackhole/19/221977/train_val_split_data/")
# PT_PATH = DATA_DIR / "preprocessed_train_val_3000hvg.pt"
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
NUM_EPOCHS = 50

# optimizer / model
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIMS = [1024, 524, 256, 128]
DROPOUT = 0.5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENE_ID_COL = "gene_name"
ISOFORM_ID_COL = "transcript_id"

# -----------------------------
# Main pipeline
# -----------------------------

def main_mlp():
    set_seed(SEED)

    genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids = load_geneaware(
        GENE_PATH,
        ISOFORM_PATH,
        max_samples=MAX_SAMPLES,
        seed=SEED,
        selection_method='top_counts'
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
        use_amp=True,
        # iso_to_gene_index=iso_to_gene_index,
    )

    print(history)

    # per-run output folder
    mlp_out = OUT_DIR / "MLP_GeneAware_45k"
    mlp_out.mkdir(parents=True, exist_ok=True)
    
    
    checkpoint_path = mlp_out / "MLP_GeneAware_45k_weights.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dims": HIDDEN_DIMS, "max_samples": MAX_SAMPLES},
            "history": history,
        },
        checkpoint_path,
    )
    
    print(f"Saved MLP checkpoint to {checkpoint_path}")
    
    plot_history(history, mlp_out / "MLP_GeneAware_45k_Training.png")

if __name__ == "__main__":
    main_mlp()