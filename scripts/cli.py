"""
Command-line entrypoint to run standard or gene-aware experiments.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.data import load_standard, load_geneaware
from scripts.models import MLP, GeneAwareMLP, VAE
from scripts.train import train_model, train_vae
from scripts.utils import set_seed, pearson_corr, EarlyStopping
from scripts.plots import plot_history, plot_residuals


def run_standard(args):
    set_seed(args.seed)
    genes_t, iso_t = load_standard(
        Path(args.genes_path),
        Path(args.iso_path),
        max_samples=args.max_samples,
        test_count=args.test_count,
        seed=args.seed,
    )
    n_samples, n_genes = genes_t.shape
    n_iso = iso_t.shape[1]

    perm = torch.randperm(n_samples)
    if args.max_samples is None:
        test_count = int(n_samples * args.test_percent)
        test_idx = perm[:test_count]
        train_idx = perm[test_count:]
    else:
        take = min(args.max_samples, n_samples)
        test_idx = perm[: min(args.test_count, take)]
        train_idx = perm[min(args.test_count, take) : take]

    train_ds = TensorDataset(genes_t[train_idx], iso_t[train_idx])
    val_ds = TensorDataset(genes_t[test_idx], iso_t[test_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(n_genes, args.hidden_dims, n_iso, dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
    loss_fn = nn.MSELoss()

    history = train_model(model, train_loader, val_loader, opt, loss_fn, args.epochs, scheduler=sched, early_stopper=stopper, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, out_dir / "train_val_plot.png")

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inp, tgt in val_loader:
            inp = inp.to(args.device).float()
            tgt = tgt.to(args.device).float()
            preds.append(model(inp).cpu())
            targets.append(tgt.cpu())
    preds = torch.cat(preds).flatten()
    targets = torch.cat(targets).flatten()
    plot_residuals(preds, targets, out_dir / "residuals_plot.png")
    print(f"Validation RMSE: {torch.sqrt(torch.mean((preds - targets) ** 2)):.4f}")
    print(f"Validation Pearson: {pearson_corr(preds, targets):.4f}")
    torch.save(model.state_dict(), out_dir / "Standard_MLP_state_dict.pth")


def run_geneaware(args):
    set_seed(args.seed)
    genes_tensor, isoform_props, gene_to_iso_map, iso_to_gene_index, isoform_ids = load_geneaware(
        Path(args.genes_path),
        Path(args.iso_path),
        min_counts=args.min_counts,
        max_genes=args.max_genes,
        max_isoforms=args.max_isoforms,
        max_samples=args.max_samples,
        load_chunk_size=args.load_chunk_size,
        count_chunk_size=args.count_chunk_size,
        proportion_batch=args.proportion_batch,
        gene_id_col=args.gene_id_col,
        isoform_id_col=args.isoform_id_col,
        seed=args.seed,
    )

    perm = torch.randperm(len(genes_tensor))
    val_size = max(1, int(len(genes_tensor) * args.val_fraction))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_ds = TensorDataset(genes_tensor[train_idx], isoform_props[train_idx])
    val_ds = TensorDataset(genes_tensor[val_idx], isoform_props[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GeneAwareMLP(
        input_dim=genes_tensor.shape[1],
        hidden_dims=args.hidden_dims,
        isoform_dim=isoform_props.shape[1],
        gene_index_per_iso=torch.from_numpy(iso_to_gene_index),
        dropout=args.dropout,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
    loss_fn = nn.MSELoss()

    history = train_model(model, train_loader, val_loader, opt, loss_fn, args.epochs, scheduler=sched, early_stopper=stopper, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, out_dir / "GeneAware_Training.png")

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inp, tgt in val_loader:
            inp = inp.to(args.device).float()
            tgt = tgt.to(args.device).float()
            preds.append(model(inp).cpu())
            targets.append(tgt.cpu())
    preds = torch.cat(preds).flatten()
    targets = torch.cat(targets).flatten()
    plot_residuals(preds, targets, out_dir / "GeneAware_Evaluation.png")
    print(f"Validation RMSE: {torch.sqrt(torch.mean((preds - targets) ** 2)):.4f}")
    print(f"Validation Pearson: {pearson_corr(preds, targets):.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "history": history,
            "gene_to_iso_map": gene_to_iso_map,
            "isoform_ids": isoform_ids,
        },
        out_dir / "GeneAware_Full_MLP_weights.pt",
    )


def run_vae(args):
    set_seed(args.seed)
    genes_t, _ = load_standard(
        Path(args.genes_path),
        Path(args.iso_path),
        max_samples=args.max_samples,
        test_count=args.test_count,
        seed=args.seed,
    )
    n_samples, n_genes = genes_t.shape

    perm = torch.randperm(n_samples)
    val_count = max(1, int(n_samples * args.test_percent))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]

    train_ds = TensorDataset(genes_t[train_idx])
    val_ds = TensorDataset(genes_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = VAE(input_dim=n_genes, latent_dim=args.latent_dim, hidden_dims=args.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train_vae(
        model,
        train_loader,
        val_loader,
        optimizer,
        device=torch.device(args.device),
        num_epochs=args.epochs,
        beta_max=args.beta,
        warmup_epochs=args.warmup_epochs,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = out_dir / "vae_scVI_results.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Training completed. Weights saved to: {ckpt_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run standard or gene-aware experiments.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    std = subparsers.add_parser("standard", help="Standard MLP")
    std.add_argument("--genes-path", required=True)
    std.add_argument("--iso-path", required=True)
    std.add_argument("--out-dir", required=True)
    std.add_argument("--max-samples", type=int, default=None)
    std.add_argument("--test-count", type=int, default=500)
    std.add_argument("--test-percent", type=float, default=0.1)
    std.add_argument("--batch-size", type=int, default=64)
    std.add_argument("--epochs", type=int, default=100)
    std.add_argument("--lr", type=float, default=1e-3)
    std.add_argument("--weight-decay", type=float, default=1e-4)
    std.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 1024, 524, 256, 128])
    std.add_argument("--dropout", type=float, default=0.25)
    std.add_argument("--seed", type=int, default=42)
    std.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    std.set_defaults(func=run_standard)

    ga = subparsers.add_parser("geneaware", help="Gene-aware isoform proportions")
    ga.add_argument("--genes-path", required=True)
    ga.add_argument("--iso-path", required=True)
    ga.add_argument("--out-dir", required=True)
    ga.add_argument("--min-counts", type=int, default=20)
    ga.add_argument("--max-genes", type=int, default=None)
    ga.add_argument("--max-isoforms", type=int, default=None)
    ga.add_argument("--max-samples", type=int, default=None)
    ga.add_argument("--val-fraction", type=float, default=0.1)
    ga.add_argument("--batch-size", type=int, default=64)
    ga.add_argument("--epochs", type=int, default=50)
    ga.add_argument("--lr", type=float, default=3e-4)
    ga.add_argument("--weight-decay", type=float, default=1e-5)
    ga.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 512, 256, 128])
    ga.add_argument("--dropout", type=float, default=0.25)
    ga.add_argument("--seed", type=int, default=42)
    ga.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ga.add_argument("--gene-id-col", default="gene_name")
    ga.add_argument("--isoform-id-col", default="transcript_id")
    ga.add_argument("--proportion-batch", type=int, default=64)
    ga.add_argument("--load-chunk-size", type=int, default=64)
    ga.add_argument("--count-chunk-size", type=int, default=128)
    ga.set_defaults(func=run_geneaware)

    vae = subparsers.add_parser("vae", help="Train VAE model")
    vae.add_argument("--genes-path", required=True)
    vae.add_argument("--iso-path", required=True)
    vae.add_argument("--out-dir", required=True)
    vae.add_argument("--max-samples", type=int, default=None)
    vae.add_argument("--test-count", type=int, default=100)
    vae.add_argument("--test-percent", type=float, default=0.1)
    vae.add_argument("--batch-size", type=int, default=64)
    vae.add_argument("--epochs", type=int, default=5)
    vae.add_argument("--lr", type=float, default=1e-3)
    vae.add_argument("--latent-dim", type=int, default=512)
    vae.add_argument("--hidden-dims", type=int, nargs="+", default=[4096, 2048, 1024])
    vae.add_argument("--beta", type=float, default=0.2)
    vae.add_argument("--warmup-epochs", type=int, default=3)
    vae.add_argument("--seed", type=int, default=42)
    vae.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    vae.set_defaults(func=run_vae)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
