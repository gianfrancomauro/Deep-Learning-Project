import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.plots import plot_history, plot_residuals
from scripts.utils import set_seed, pearson_corr, classification_error, EarlyStopping
from scripts.train import train_LSTM
from scripts.data import load_standard_hvg
from scripts.models import LSTMIsoformer  

plt.rcParams["figure.dpi"] = 120
torch.set_float32_matmul_precision("high")


DATA_DIR = Path("/dtu/blackhole/19/221977/train_val_split_data/")
TRAIN_PT = DATA_DIR / "preprocessed_train_3000hvg_train.pt"
TEST_PT  = DATA_DIR / "preprocessed_train_3000hvg_test.pt"

OUT_DIR = Path("Results/LSTM_20epochs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = None
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 20
VAL_SET_PERCENT = 0.1

LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Main pipeline
def main():

    set_seed(SEED)
    t0 = time.time()
    print("Loading preprocessed train/val data...")
    genes_t, iso_t, gene_to_iso_map, iso_to_gene, isoform_ids, gene_ids = load_standard_hvg(TRAIN_PT)

    if MAX_SAMPLES is not None and MAX_SAMPLES < len(genes_t):
        rng = np.random.default_rng(SEED)
        subset_idx = rng.choice(len(genes_t), size=MAX_SAMPLES, replace=False)
        subset_idx.sort()
        genes_t = genes_t[subset_idx]
        iso_t = iso_t[subset_idx]
        print(f"Subsampled to {len(genes_t)} samples (MAX_SAMPLES={MAX_SAMPLES})")

    n_samples = len(genes_t)
    n_genes = genes_t.shape[1]
    n_iso = iso_t.shape[1]
    
    print(f"Train/val data loaded in {time.time()-t0:.2f}s: n_samples={n_samples}, n_genes={n_genes}, n_isoforms={n_iso}")

    # Convert mapping to tensor
    iso_to_gene_idx = torch.from_numpy(iso_to_gene).long().to(DEVICE)

    # shuffle and split
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(n_samples)
    
    # If MAX_SAMPLES was used, n_samples is already reduced. We just split what we have.
    val_count = int(n_samples * VAL_SET_PERCENT)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_dataset = TensorDataset(genes_t[train_idx], iso_t[train_idx])
    val_dataset   = TensorDataset(genes_t[val_idx],  iso_t[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading preprocessed held-out test data...")
    test_genes, test_iso, _, test_iso_to_gene, _, _ = load_standard_hvg(TEST_PT)
    if test_genes.shape[1] != n_genes or test_iso.shape[1] != n_iso:
        raise ValueError(
            f"Test features mismatch: train (genes={n_genes}, iso={n_iso}) vs test (genes={test_genes.shape[1]}, iso={test_iso.shape[1]})"
        )
    if not np.array_equal(test_iso_to_gene, iso_to_gene):
        raise ValueError("iso_to_gene_index mismatch between train and test preprocessed files.")
    test_dataset = TensorDataset(test_genes, test_iso)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------

    model = LSTMIsoformer(
        input_dim=n_genes,
        isoform_dim=n_iso,
        gene_index_per_iso=iso_to_gene_idx,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,  
    ).to(DEVICE)

    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    
    early_stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")

    print("Training...")

    history = train_LSTM(
        model, train_loader, val_loader, optimizer, loss_fn, NUM_EPOCHS, scheduler, 
        early_stopper=early_stopper,
        device=DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        iso_to_gene_index=iso_to_gene_idx
    )
    
    train_losses = history["train_loss"]
    train_corrs = history["train_corr"]
    val_losses = history["val_loss"]
    val_corrs = history["val_corr"]

    print("Training finished.")
    
    # Plot history
    plot_history(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_corr": train_corrs,
            "val_corr": val_corrs
        },
        OUT_DIR / "training_history.png"
    )

    final_val_loss = val_losses[-1]
    final_val_corr = val_corrs[-1]

    # compute final validation classification error
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE).float()
            targets = targets.to(DEVICE).float()
            outputs = model(inputs)
            preds_list.append(outputs.cpu())
            targets_list.append(targets.cpu())

    preds = torch.cat(preds_list).to(DEVICE)
    targets = torch.cat(targets_list).to(DEVICE)

    final_val_class_error = classification_error(preds, targets, iso_to_gene_idx)

    # Evaluate on held-out test set
    test_loss_acc = 0.0
    test_corr_acc = 0.0
    test_total = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE).float()
            targets = targets.to(DEVICE).float()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_size = inputs.size(0)
            test_loss_acc += loss.item() * batch_size
            test_corr_acc += pearson_corr(outputs.flatten(), targets.flatten()).item() * batch_size
            test_total += batch_size
            test_preds.append(outputs.cpu())
            test_targets.append(targets.cpu())

    test_loss = test_loss_acc / max(1, test_total)
    test_corr = test_corr_acc / max(1, test_total)
    test_preds_t = torch.cat(test_preds)
    test_targets_t = torch.cat(test_targets)
    test_class_error = classification_error(test_preds_t, test_targets_t, iso_to_gene_idx)
    
    # Plot test residuals
    plot_residuals(test_preds_t, test_targets_t, OUT_DIR / "test_residuals.png")

    # Save single-value metrics
    with open(OUT_DIR / "final_metrics.txt", "w") as f:
        f.write(f"Final validation loss: {final_val_loss:.6f}\n")
        f.write(f"Final validation Pearson: {final_val_corr:.6f}\n")
        f.write(f"Final validation classification error: {final_val_class_error:.6f}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test Pearson: {test_corr:.6f}\n")
        f.write(f"Test classification error: {test_class_error:.6f}\n")

    # Save full curves (all epochs)
    np.save(OUT_DIR / "train_losses.npy", np.array(train_losses))
    np.save(OUT_DIR / "val_losses.npy", np.array(val_losses))
    np.save(OUT_DIR / "train_corrs.npy", np.array(train_corrs))
    np.save(OUT_DIR / "val_corrs.npy", np.array(val_corrs))

    print(f"Saved metrics and plots to {OUT_DIR}")

    # Save model weights
    weights_path = OUT_DIR / "lstm_isoformer_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

if __name__ == "__main__":
    main()
