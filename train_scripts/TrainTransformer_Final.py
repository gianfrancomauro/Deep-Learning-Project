from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from scripts.data import load_geneaware
from scripts.models import TransformerIsoformer
from scripts.train import train_transformer
from scripts.utils import set_seed, pearson_corr, EarlyStopping, classification_error
from scripts.plots import plot_history, plot_residuals
import mlflow

plt.rcParams["figure.dpi"] = 120
torch.set_float32_matmul_precision("high")

# Configuration
DATA_DIR = Path("/zhome/af/a/221977/Blackhole/train_val_split_data/")
GENE_PATH = DATA_DIR / "sc_processed_genes_train_val_set.h5ad"
ISOFORM_PATH = DATA_DIR / "sc_processed_transcripts_train_val_set.h5ad"
OUTPUT_DIR = Path("Results/Transformer_amp")


MIN_COUNTS = 20
MAX_GENES = 3000  # Top HVGs
MAX_ISOFORMS = None
MAX_SAMPLES = None # Full dataset
VAL_FRACTION = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 25
LR = 1e-4 # lower LR for Transformer
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformer Hyperparameters
D_MODEL = 256
N_HEAD = 4
N_LAYERS = 2
D_FF = 1024

GENE_ID_COL = "gene_name"
ISOFORM_ID_COL = "transcript_id"

mlflow.set_experiment("Transformer_Isoformer_Training")
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    print(f"Loading data from {DATA_DIR}...")
    
    # Load data into memory with filtering
    genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids = load_geneaware(
        gene_path=GENE_PATH,
        isoform_path=ISOFORM_PATH,
        min_counts=MIN_COUNTS,
        max_genes=MAX_GENES,
        max_isoforms=MAX_ISOFORMS,
        max_samples=MAX_SAMPLES,
        # genes var lacks identifiers; use index on both sides explicitly
        gene_id_col=None,
        isoform_id_col=None,
        seed=SEED,
        selection_method="hvg"
    )
    
    n_genes = genes_tensor.shape[1]
    n_isoforms = isoform_proportions.shape[1]
    iso_to_gene_index_tensor = torch.from_numpy(iso_to_gene_index).long()

    print(f"Total samples: {len(genes_tensor)}")
    print(f"Genes: {n_genes}")
    print(f"Isoforms: {n_isoforms}")

    full_dataset = TensorDataset(genes_tensor, isoform_proportions)

    # Split train/val
    n_samples = len(full_dataset)
    val_size = max(1, int(n_samples * VAL_FRACTION))
    train_size = n_samples - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # In-memory dataset is fast, so num_workers can be 0 or small
    pin_memory = DEVICE.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    print("Initializing TransformerIsoformer...")
    model = TransformerIsoformer(
        input_dim=1, # Treat each gene as a token with 1 feature
        isoform_dim=n_isoforms,
        gene_index_per_iso=iso_to_gene_index_tensor,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
    loss_fn = nn.MSELoss()

    params = {
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "min_counts": MIN_COUNTS,
        "max_genes": MAX_GENES,
        "max_isoforms": MAX_ISOFORMS,
        "max_samples": MAX_SAMPLES,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        history = train_transformer(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            NUM_EPOCHS,
            scheduler=scheduler,
            early_stopper=stopper,
            device=DEVICE,
            iso_to_gene_index=iso_to_gene_index_tensor,
        )

        plot_history(history, OUTPUT_DIR / "Transformer_Training.png")
        mlflow.log_artifact(OUTPUT_DIR / "Transformer_Training.png")

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
        class_err = classification_error(preds, targets, iso_to_gene_index_tensor)
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation Pearson: {corr:.4f}")
        print(f"Validation classification error: {class_err:.4f}")
        
        mlflow.log_metrics({
            "final_val_rmse": rmse.item(),
            "final_val_pearson": corr.item(),
            "final_val_class_error": class_err,
        })

        plot_residuals(preds, targets, OUTPUT_DIR / "Transformer_Evaluation.png")
        mlflow.log_artifact(OUTPUT_DIR / "Transformer_Evaluation.png")

        checkpoint_path = OUTPUT_DIR / "Transformer_weights.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": params,
                "history": history,
                "sorted_gene_ids": list(gene_to_iso_map.keys()),
                "gene_to_iso_map": gene_to_iso_map,
                "isoform_ids": isoform_ids,
                "iso_to_gene_index": iso_to_gene_index,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
        # mlflow.log_artifact(checkpoint_path)


if __name__ == "__main__":
    main()
