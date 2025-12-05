"""
Shared utilities: seeding, metrics, early stopping.
"""
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    vx = x - x.mean()
    vy = y - y.mean()
    return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)


def classification_error(preds: torch.Tensor, targets: torch.Tensor, iso_to_gene_index: torch.Tensor) -> float:
    """
    Per-gene top-isoform error: for each gene, compare the predicted top isoform
    to the true top isoform. Returns mean error over samples and genes.
    """
    iso_to_gene_index = iso_to_gene_index.to(preds.device)
    n_genes = int(iso_to_gene_index.max().item()) + 1
    batch_size = preds.size(0)

    def top_iso_per_gene(mat):
        # mat: [batch, n_iso]
        # Returns: [batch, n_genes] with argmax isoform index per gene

        max_vals = torch.full((batch_size, n_genes), -1e9, device=mat.device, dtype=mat.dtype)
        argmax_indices = torch.full((batch_size, n_genes), -1, device=mat.device, dtype=torch.long)

        idx_expanded = iso_to_gene_index.unsqueeze(0).expand(batch_size, -1)

        # Scatter reduce to find max value per gene
        max_vals.scatter_reduce_(1, idx_expanded, mat, reduce="amax", include_self=False)

        # Find which isoforms match the max for their gene
        gene_max_at_iso = max_vals.gather(1, idx_expanded)
        is_max = (mat == gene_max_at_iso)

        # Among max isoforms, pick the one with smallest index (tie-breaking)
        iso_indices = torch.arange(mat.size(1), device=mat.device).unsqueeze(0).expand(batch_size, -1)
        masked_indices = torch.where(is_max, iso_indices, torch.full_like(iso_indices, mat.size(1)))

        # Scatter min to get the smallest isoform index per gene among the maxes
        argmax_indices.scatter_reduce_(1, idx_expanded, masked_indices, reduce="amin", include_self=False)

        return argmax_indices

    pred_top = top_iso_per_gene(preds)
    target_top = top_iso_per_gene(targets)
    return (pred_top != target_top).float().mean().item()



    


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False
        improvement = (
            current_score < self.best_score - self.min_delta
            if self.mode == "min"
            else current_score > self.best_score + self.min_delta
        )
        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop