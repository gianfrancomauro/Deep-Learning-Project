"""
Data loading and preprocessing helpers.
"""
from pathlib import Path
from typing import Dict, List, Sequence
import anndata as ad
import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import gc
from sklearn.decomposition import PCA


class GenesDataset(torch.utils.data.Dataset):
    """
    Minimal dataset wrapper for gene expression inputs and isoform targets.
    Used by VAE/MLP pipelines that expect paired tensors.
    """

    def __init__(self, genes_X, isoforms_Y):
        if not torch.is_tensor(genes_X):
            genes_X = torch.tensor(genes_X)
        if not torch.is_tensor(isoforms_Y):
            isoforms_Y = torch.tensor(isoforms_Y)
        if genes_X.shape[0] != isoforms_Y.shape[0]:
            raise ValueError(
                f"genes_X and isoforms_Y must share sample count; got {genes_X.shape[0]} vs {isoforms_Y.shape[0]}"
            )
        self.genes_X = genes_X.clone().float()
        self.isoforms_Y = isoforms_Y.clone().float()

    def __len__(self):
        return self.genes_X.size(0)

    def __getitem__(self, idx):
        return self.genes_X[idx], self.isoforms_Y[idx]


def select_sample_indices(n_obs: int, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or max_samples >= n_obs:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_obs, size=max_samples, replace=False)
    idx.sort()
    return idx


def compute_feature_mask(backed_adata: ad.AnnData, min_counts: int, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    totals = np.zeros(backed_adata.n_vars, dtype=np.float64)
    matrix = backed_adata.X
    for start in range(0, backed_adata.n_obs, chunk_size):
        stop = min(backed_adata.n_obs, start + chunk_size)
        batch = matrix[start:stop]
        if sp.issparse(batch):
            batch_totals = np.asarray(batch.sum(axis=0)).ravel()
        else:
            batch_totals = np.asarray(batch).sum(axis=0)
        totals += batch_totals
    mask = totals >= min_counts
    return mask, totals


def apply_topk(mask: np.ndarray, totals: np.ndarray, max_features: int | None) -> np.ndarray:
    if max_features is None or mask.sum() <= max_features:
        return mask
    candidate_idx = np.where(mask)[0]
    order = np.argsort(totals[candidate_idx])[::-1]
    keep = candidate_idx[order[:max_features]]
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[keep] = True
    return new_mask


def load_dense_subset(backed_adata: ad.AnnData, obs_idx: Sequence[int], var_mask: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    obs_idx = np.asarray(obs_idx)
    if obs_idx.ndim != 1:
        obs_idx = obs_idx.ravel()
    if var_mask.dtype == bool:
        var_idx = np.flatnonzero(var_mask)
    else:
        var_idx = np.asarray(var_mask)
    n_obs = len(obs_idx)
    n_vars = len(var_idx)
    dense = np.empty((n_obs, n_vars), dtype=np.float32)
    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        rows = obs_idx[start:stop]
        if not len(rows):
            continue
        chunk_view = backed_adata[rows, :]
        chunk = chunk_view.X
        if sp.issparse(chunk):
            chunk = chunk[:, var_idx].toarray()
        else:
            chunk = np.asarray(chunk)
            chunk = chunk[:, var_idx]
        dense[start:stop] = chunk.astype(np.float32, copy=False)
    return dense


def build_gene_isoform_mapping(var_df, gene_col: str | None, isoform_col: str | None = None) -> tuple[List[str], Dict[str, List[int]], np.ndarray, List[str]]:
    # Fallbacks: if a requested column is missing, use the index to avoid hard crashes.
    if gene_col and gene_col in var_df.columns:
        gene_series = var_df[gene_col].astype(str).fillna("unknown_gene")
    else:
        gene_series = var_df.index.to_series().astype(str).fillna("unknown_gene")

    if isoform_col and isoform_col in var_df.columns:
        isoform_ids = var_df[isoform_col].astype(str).fillna(var_df.index.astype(str)).tolist()
    else:
        isoform_ids = var_df.index.astype(str).tolist()

    gene_to_index: Dict[str, int] = {}
    gene_to_iso: Dict[str, List[int]] = {}
    sorted_genes: List[str] = []
    iso_to_gene_index = np.empty(len(var_df), dtype=np.int64)

    for iso_idx, gene_id in enumerate(gene_series.values):
        if gene_id not in gene_to_index:
            gene_to_index[gene_id] = len(sorted_genes)
            sorted_genes.append(gene_id)
            gene_to_iso[gene_id] = []
        g_idx = gene_to_index[gene_id]
        gene_to_iso[gene_id].append(iso_idx)
        iso_to_gene_index[iso_idx] = g_idx

    return sorted_genes, gene_to_iso, iso_to_gene_index, isoform_ids


def counts_to_gene_proportions(
    counts: torch.Tensor,
    iso_to_gene_index: np.ndarray,
    n_genes: int,
    chunk_size: int = 128,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    idx_tensor = torch.from_numpy(iso_to_gene_index).long().to(device=counts.device)
    if out is None:
        proportions = torch.empty_like(counts)
    else:
        proportions = out

    max_chunk = min(chunk_size, counts.shape[0])
    tmp_totals = torch.empty((max_chunk, n_genes), dtype=counts.dtype, device=counts.device)
    idx_rows = idx_tensor.unsqueeze(0)

    for start in range(0, counts.shape[0], chunk_size):
        end = min(start + chunk_size, counts.shape[0])
        batch = counts[start:end]
        batch_size = end - start
        expand_idx = idx_rows.expand(batch_size, -1)
        totals = tmp_totals[:batch_size]
        totals.zero_()
        totals.scatter_add_(1, expand_idx, batch)
        denom = torch.gather(totals, 1, expand_idx).clamp_min_(1e-8)
        proportions[start:end].copy_(batch / denom)

    return proportions


def normalize_and_log(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def load_standard_hvg(
    pt_path: Path,
    max_samples: int | None = None,
    test_count: int = 500,
    seed: int = 42,
    compute_pca: bool = False,
    n_comps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, List[int]], np.ndarray, List[str], List[str]]:


    pt_dict = torch.load(pt_path, weights_only=False)


    genes_tensor = pt_dict['genes_tensor']
    isoform_proportions = pt_dict['isoform_proportions']
    gene_to_iso_map = pt_dict['gene_to_iso_map']  # shared mapping
    iso_to_gene_index = pt_dict['iso_to_gene_index']  # shared mapping
    isoform_ids = pt_dict['isoform_ids']  # shared ids
    gene_ids = pt_dict['gene_ids']

    """
    def read_X_subset(path: Path, idx: np.ndarray) -> np.ndarray:
        with h5py.File(path, "r") as f:
            if isinstance(f["X"], h5py.Group):
                # Sparse matrix (CSR)
                g = f["X"]
                # Read indptr to find row boundaries
                indptr = g["indptr"][:]
                
                # We need to read specific rows. 
                # For each row i in idx, data is in range [indptr[i], indptr[i+1])
                # Since idx is sorted, we can iterate and read chunks?
                # Or just read row by row.
                
                n_rows = len(idx)
                n_cols = g.attrs["shape"][1]
                out = np.zeros((n_rows, n_cols), dtype=np.float32)
                
                # To avoid many small reads, we could group consecutive indices?
                # For now, let's just do row by row but be careful not to read whole arrays.
                
                # We need to access 'data' and 'indices' datasets.
                ds_data = g["data"]
                ds_indices = g["indices"]
                
                for i, row_idx in enumerate(idx):
                    start = indptr[row_idx]
                    end = indptr[row_idx + 1]
                    if start == end:
                        continue
                    
                    # Read only the slice for this row
                    row_data = ds_data[start:end]
                    row_indices = ds_indices[start:end]
                    
                    out[i, row_indices] = row_data
                    
                return out
            else:
                return f["X"][idx]

    # Get total samples from one file (assuming they match)
    with h5py.File(genes_path, "r") as f:
        if isinstance(f["X"], h5py.Group):
            n_samples = f["X"].attrs["shape"][0]
        else:
            n_samples = f["X"].shape[0]

    
    if max_samples is not None and max_samples < n_samples:
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_samples)

    print(f"Loading {len(idx)} samples from {genes_path}...")
    genes_arr = read_X_subset(genes_path, idx)
    
    print(f"Loading {len(idx)} samples from {iso_path}...")
    iso_arr = read_X_subset(iso_path, idx)
    
    print("Normalizing...")
    
    def normalize_log1p(X):
        # X is numpy array [n_obs, n_vars]
        counts_per_cell = X.sum(axis=1, keepdims=True)
        # Avoid division by zero
        counts_per_cell[counts_per_cell == 0] = 1
        X_norm = X / counts_per_cell * 1e4
        return np.log1p(X_norm)

    genes_arr = normalize_log1p(genes_arr)
    iso_arr = normalize_log1p(iso_arr)

    genes_arr = np.nan_to_num(genes_arr, nan=0.0)
    iso_arr = np.nan_to_num(iso_arr, nan=0.0)


    # Shuffle for training
    """    
    rng = np.random.default_rng(seed)
    shuffle_idx = rng.permutation(genes_tensor.shape[0])
    genes_tensor = genes_tensor[shuffle_idx]
    isoform_proportions = isoform_proportions[shuffle_idx]

    if compute_pca:
        # Compute PCA using sklearn

        print("Computing PCA...")
        pca = PCA(n_components=n_comps, svd_solver='auto', random_state=0)
        genes_pca = pca.fit_transform(genes_tensor.numpy())
        genes_tensor = torch.from_numpy(genes_pca).float()

    return genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids, gene_ids

def load_scGPT(
    genes_path: Path,
    iso_path: Path,
    max_samples: int | None = None,
    test_count: int = 500,
    seed: int = 42,
    compute_pca: bool = False,
    n_comps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:

    def read_X_subset(path: Path, idx: np.ndarray) -> np.ndarray:
        with h5py.File(path, "r") as f:
            if isinstance(f["X"], h5py.Group):
                # Sparse matrix (CSR)
                g = f["X"]
                # Read indptr to find row boundaries
                indptr = g["indptr"][:]
                n_rows = len(idx)
                n_cols = g.attrs["shape"][1]
                out = np.zeros((n_rows, n_cols), dtype=np.float32)
                ds_data = g["data"]
                ds_indices = g["indices"]
                for i, row_idx in enumerate(idx):
                    start = indptr[row_idx]
                    end = indptr[row_idx + 1]
                    if start == end:
                        continue
                    row_data = ds_data[start:end]
                    row_indices = ds_indices[start:end]
                    out[i, row_indices] = row_data
                return out
            else:
                return f["X"][idx]

    # Get total samples from one file (assuming they match)
    with h5py.File(genes_path, "r") as f:
        if isinstance(f["X"], h5py.Group):
            n_samples = f["X"].attrs["shape"][0]
        else:
            n_samples = f["X"].shape[0]

    rng = np.random.default_rng(seed)
    if max_samples is not None and max_samples < n_samples:
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_samples)

    print(f"Loading {len(idx)} samples from {genes_path}...")

    adata_emb = ad.read_h5ad(genes_path)[idx]
    genes_arr = np.array(adata_emb.obsm['X_scGPT'])
    del adata_emb
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading {len(idx)} samples from {iso_path}...")
    iso_arr = read_X_subset(iso_path, idx)
    
    print("Normalizing isoform counts...")
    counts_per_cell = iso_arr.sum(axis=1, keepdims=True)
    counts_per_cell[counts_per_cell == 0] = 1
    iso_arr = iso_arr / counts_per_cell * 1e4
    iso_arr = np.log1p(iso_arr)

    genes_arr = np.nan_to_num(genes_arr, nan=0.0)
    iso_arr = np.nan_to_num(iso_arr, nan=0.0)

    if compute_pca:
        print("Computing PCA...")
        adata_genes = ad.AnnData(genes_arr)
        sc.pp.pca(adata_genes, n_comps=n_comps, svd_solver='auto', zero_center=True, random_state=0)
        genes_arr = adata_genes.obsm["X_pca"]

    # Shuffle for training
    shuffle_idx = rng.permutation(len(genes_arr))
    genes_t = torch.from_numpy(genes_arr[shuffle_idx]).float()
    iso_t = torch.from_numpy(iso_arr[shuffle_idx]).float()
    
    return genes_t, iso_t

def load_geneaware(
    gene_path: Path,
    isoform_path: Path,
    min_counts: int = 20,
    max_genes: int | None = 3000,
    max_isoforms: int | None = None,
    max_samples: int | None = None,
    load_chunk_size: int = 64,
    count_chunk_size: int = 128,
    proportion_batch: int = 64,
    gene_id_col: str = "gene_name",
    isoform_id_col: str = "transcript_id",
    seed: int = 42,
    selection_method: str = "hvg", # "top_counts" or "hvg"
    fixed_gene_ids: Sequence[str] | None = None,
    fixed_isoform_ids: Sequence[str] | None = None,
    return_gene_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, List[int]], np.ndarray, List[str]] | tuple[torch.Tensor, torch.Tensor, Dict[str, List[int]], np.ndarray, List[str], List[str]]:
    
    genes_backed = ad.read_h5ad(gene_path.as_posix(), backed="r")
    isoforms_backed = ad.read_h5ad(isoform_path.as_posix(), backed="r")
    sample_idx = select_sample_indices(genes_backed.n_obs, max_samples, seed)

    # -----------------------------
    # Gene feature selection / alignment
    # -----------------------------
    gene_id_series = (
        genes_backed.var[gene_id_col].astype(str)
        if gene_id_col and gene_id_col in genes_backed.var.columns
        else genes_backed.var.index.astype(str)
    )
    if gene_id_col and gene_id_col not in genes_backed.var.columns:
        print(f"[data] gene_id_col '{gene_id_col}' not found in genes var; using index instead.")

    if fixed_gene_ids is not None:
        # Align to provided gene IDs (e.g., preprocessing test set to match train features)
        gene_id_to_idx = {gid: i for i, gid in enumerate(gene_id_series.values)}
        selected_indices = []
        missing_genes = []
        for gid in fixed_gene_ids:
            if gid in gene_id_to_idx:
                selected_indices.append(gene_id_to_idx[gid])
            else:
                missing_genes.append(gid)
        if missing_genes:
            print(f"[data] Warning: {len(missing_genes)} genes from fixed list missing in dataset.")
        gene_mask = np.zeros(len(gene_id_series), dtype=bool)
        gene_mask[selected_indices] = True
        gene_var_indices = np.array(selected_indices, dtype=int)
    else:
        gene_mask, gene_totals = compute_feature_mask(genes_backed, min_counts, chunk_size=count_chunk_size)
        if selection_method == "hvg":
            print("Selecting genes using HVG...")
            # Backed AnnData cannot be copied without a filename; load subset densely instead
            tmp_X = load_dense_subset(genes_backed, sample_idx, gene_mask, chunk_size=load_chunk_size)
            tmp_adata = ad.AnnData(tmp_X, var=genes_backed.var[gene_mask].copy())
            sc.pp.normalize_total(tmp_adata, target_sum=1e4)
            sc.pp.log1p(tmp_adata)
            n_hvg = min(max_genes or gene_mask.sum(), gene_mask.sum())
            sc.pp.highly_variable_genes(
                tmp_adata,
                n_top_genes=n_hvg,
                subset=False,
                flavor="seurat"
            )
            hvg_subset_mask = tmp_adata.var["highly_variable"].values
            valid_indices = np.flatnonzero(gene_mask)
            hvg_indices = valid_indices[hvg_subset_mask]
            new_mask = np.zeros_like(gene_mask)
            new_mask[hvg_indices] = True
            gene_mask = new_mask
            del tmp_adata, tmp_X
        else:
            gene_mask = apply_topk(gene_mask, gene_totals, max_genes)
        gene_var_indices = np.flatnonzero(gene_mask)
    
    # Get selected gene IDs (flexible column) in the order used for tensors
    genes_var = genes_backed.var.iloc[gene_var_indices].copy()
    gene_ids_arr = (
        genes_var[gene_id_col].astype(str).values
        if gene_id_col and gene_id_col in genes_var.columns
        else genes_var.index.astype(str).values
    )
    selected_genes = set(gene_ids_arr)
    
    # Filter isoforms: min_counts AND belonging to selected genes (or align to provided IDs)
    iso_var_all = isoforms_backed.var
    iso_gene_col = None
    if gene_id_col and gene_id_col in iso_var_all.columns:
        iso_gene_col = gene_id_col
    elif "gene_name" in iso_var_all.columns:
        iso_gene_col = "gene_name"

    if fixed_isoform_ids is not None:
        iso_id_series = (
            iso_var_all[isoform_id_col].astype(str)
            if isoform_id_col and isoform_id_col in iso_var_all.columns
            else iso_var_all.index.astype(str)
        )
        iso_id_to_idx = {iid: i for i, iid in enumerate(iso_id_series.values)}
        iso_indices = []
        missing_iso = []
        for iid in fixed_isoform_ids:
            if iid in iso_id_to_idx:
                iso_indices.append(iso_id_to_idx[iid])
            else:
                missing_iso.append(iid)
        if missing_iso:
            print(f"[data] Warning: {len(missing_iso)} isoforms from fixed list missing in dataset.")
        iso_mask = np.zeros(len(iso_id_series), dtype=bool)
        iso_mask[iso_indices] = True
        iso_var_indices = np.array(iso_indices, dtype=int)
    else:
        iso_mask, iso_totals = compute_feature_mask(isoforms_backed, min_counts, chunk_size=count_chunk_size)
        if iso_gene_col is None:
            print(f"[data] gene_id_col '{gene_id_col}' not found in isoform var; using index instead.")
            iso_gene_ids = iso_var_all.index.astype(str).values
        else:
            iso_gene_ids = iso_var_all[iso_gene_col].astype(str).values
        iso_gene_mask = np.array([gid in selected_genes for gid in iso_gene_ids])
        iso_mask = iso_mask & iso_gene_mask
        if not iso_mask.any():
            raise ValueError("No isoforms remained after gene-based filtering; check gene_id_col settings.")
        iso_mask = apply_topk(iso_mask, iso_totals, max_isoforms)
        iso_var_indices = np.flatnonzero(iso_mask)

    isoforms_var = isoforms_backed.var.iloc[iso_var_indices].copy()

    genes_np = load_dense_subset(genes_backed, sample_idx, gene_var_indices, chunk_size=load_chunk_size)
    iso_np = load_dense_subset(isoforms_backed, sample_idx, iso_var_indices, chunk_size=load_chunk_size)

    genes_adata = ad.AnnData(genes_np, var=genes_var.reset_index(drop=True))
    isoforms_adata = ad.AnnData(iso_np, var=isoforms_var.reset_index(drop=True))

    # Inputs: log-normalized genes; Targets: proportions from normalized (non-log) isoform counts
    normalize_and_log(genes_adata)
    sc.pp.normalize_total(isoforms_adata, target_sum=1e4)

    genes_np = np.nan_to_num(np.asarray(genes_adata.X), nan=0.0)
    iso_np_norm = np.nan_to_num(np.asarray(isoforms_adata.X), nan=0.0)

    genes_tensor = torch.from_numpy(genes_np)
    isoform_counts = torch.from_numpy(iso_np_norm)

    isoforms_var_reset = isoforms_var.reset_index(drop=True)
    isoform_id_col_resolved = isoform_id_col if isoform_id_col in isoforms_var_reset.columns else None
    # Prefer the gene_id_col if present, else fall back to gene_name if available
    if gene_id_col and gene_id_col in isoforms_var_reset.columns:
        isoform_gene_col = gene_id_col
    elif "gene_name" in isoforms_var_reset.columns:
        isoform_gene_col = "gene_name"
    else:
        isoform_gene_col = None
    if gene_id_col and gene_id_col not in isoforms_var_reset.columns:
        print(f"[data] gene_id_col '{gene_id_col}' not found in isoform var; using index instead.")
    if isoform_id_col and isoform_id_col not in isoforms_var_reset.columns:
        print(f"[data] isoform_id_col '{isoform_id_col}' not found in isoform var; using index instead.")

    if fixed_gene_ids is not None:
        # Align isoform->gene mapping to the fixed gene order
        isoform_ids = (
            isoforms_var_reset[isoform_id_col].astype(str).tolist()
            if isoform_id_col and isoform_id_col in isoforms_var_reset.columns
            else isoforms_var_reset.index.astype(str).tolist()
        )
        isoform_gene_ids = (
            isoforms_var_reset[isoform_gene_col].astype(str).tolist()
            if isoform_gene_col
            else isoforms_var_reset.index.astype(str).tolist()
        )
        gene_id_to_idx = {gid: i for i, gid in enumerate(gene_ids_arr)}
        iso_to_gene_index = np.empty(len(isoform_gene_ids), dtype=np.int64)
        gene_to_iso_map: Dict[str, List[int]] = {gid: [] for gid in gene_ids_arr}
        for iso_idx, gene_id in enumerate(isoform_gene_ids):
            if gene_id not in gene_id_to_idx:
                raise ValueError(f"Isoform gene '{gene_id}' not found in fixed_gene_ids.")
            g_idx = gene_id_to_idx[gene_id]
            iso_to_gene_index[iso_idx] = g_idx
            gene_to_iso_map[gene_id].append(iso_idx)
        sorted_gene_ids = list(gene_ids_arr)
    else:
        sorted_gene_ids, gene_to_iso_map, iso_to_gene_index, isoform_ids = build_gene_isoform_mapping(
            isoforms_var_reset,
            gene_col=isoform_gene_col,
            isoform_col=isoform_id_col_resolved,
        )

    n_genes = len(sorted_gene_ids)
    isoform_proportions = counts_to_gene_proportions(
        isoform_counts,
        iso_to_gene_index=iso_to_gene_index,
        n_genes=n_genes,
        chunk_size=proportion_batch,
        out=isoform_counts,
    )
    if return_gene_ids:
        return genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids, gene_ids_arr.tolist()
    return genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids

class LazyGeneAwareDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        genes_path: Path,
        isoform_path: Path,
        gene_id_col: str = "gene_name",
        isoform_id_col: str = "transcript_id",
        seed: int = 42,
        in_memory: bool = False,
        max_samples: int | None = None,
        min_counts: int = 20,
        max_genes: int | None = 3000,
        max_isoforms: int | None = None,
        selection_method: str = "hvg",  # "top_counts" or "hvg"
        count_chunk_size: int = 2048,
    ):
        super().__init__()
        self.genes_path = str(genes_path)
        self.isoform_path = str(isoform_path)
        self.in_memory = in_memory
        self.max_samples = max_samples
        self.seed = seed
        
        # 1. Setup Sampling
        print(f"Reading metadata from {self.genes_path}...")
        genes_backed = ad.read_h5ad(self.genes_path, backed="r")
        total_obs = genes_backed.n_obs
        
        rng = np.random.default_rng(seed)
        if max_samples is not None and max_samples < total_obs:
            self.sample_idx = np.sort(rng.choice(total_obs, size=max_samples, replace=False))
        else:
            self.sample_idx = np.arange(total_obs)
        
        self.n_samples = len(self.sample_idx)
        print(f"Selected {self.n_samples} samples.")

        # 2. Feature Selection (Genes)
        print("Computing gene feature mask...")
        gene_mask, gene_totals = compute_feature_mask(genes_backed, min_counts, chunk_size=count_chunk_size)
        
        if selection_method == "hvg":
            print("Selecting genes using HVG...")
            
            # Load dense subset for HVG
            # Note: We only need the genes that passed min_counts
            tmp_idx = self.sample_idx
            
            print(f"Loading temp data for HVG ({len(tmp_idx)} samples, {gene_mask.sum()} genes)...")
            tmp_X = load_dense_subset(genes_backed, tmp_idx, gene_mask, chunk_size=count_chunk_size)
            tmp_adata = ad.AnnData(tmp_X, var=genes_backed.var[gene_mask])
            
            sc.pp.normalize_total(tmp_adata, target_sum=1e4)
            sc.pp.log1p(tmp_adata)
            n_hvg = min(max_genes or gene_mask.sum(), gene_mask.sum())
            sc.pp.highly_variable_genes(
                tmp_adata,
                n_top_genes=n_hvg,
                subset=False,
                flavor="seurat"
            )
            hvg_subset_mask = tmp_adata.var["highly_variable"].values
            
            # Map back to original indices
            valid_indices = np.flatnonzero(gene_mask)
            hvg_indices = valid_indices[hvg_subset_mask]
            new_mask = np.zeros_like(gene_mask)
            new_mask[hvg_indices] = True
            self.gene_mask = new_mask
            
            del tmp_adata, tmp_X
        else:
            self.gene_mask = apply_topk(gene_mask, gene_totals, max_genes)
            
        self.gene_indices = np.flatnonzero(self.gene_mask)
        self.n_genes = len(self.gene_indices)
        print(f"Final gene count: {self.n_genes}")
        
        genes_var = genes_backed.var[self.gene_mask].copy()
        gene_ids_arr = (
            genes_var[gene_id_col].astype(str).values
            if gene_id_col and gene_id_col in genes_var.columns
            else genes_var.index.astype(str).values
        )
        selected_genes = set(gene_ids_arr)

        print("Computing isoform feature mask...")
        isoforms_backed = ad.read_h5ad(self.isoform_path, backed="r")
        iso_mask, iso_totals = compute_feature_mask(isoforms_backed, min_counts, chunk_size=count_chunk_size)

        iso_var_all = isoforms_backed.var
        iso_gene_col = None
        if gene_id_col and gene_id_col in iso_var_all.columns:
            iso_gene_col = gene_id_col
        elif "gene_name" in iso_var_all.columns:
            iso_gene_col = "gene_name"

        if iso_gene_col is None:
            iso_gene_ids = iso_var_all.index.astype(str).values
        else:
            iso_gene_ids = iso_var_all[iso_gene_col].astype(str).values
        iso_gene_mask = np.array([gid in selected_genes for gid in iso_gene_ids])
        
        iso_mask = iso_mask & iso_gene_mask
        self.iso_mask = apply_topk(iso_mask, iso_totals, max_isoforms)
        
        self.iso_indices = np.flatnonzero(self.iso_mask)
        # self.n_isoforms = len(self.iso_indices) # Wait, we need mapping first
        
        # 4. Build Mapping
        self.iso_var = isoforms_backed.var[self.iso_mask].reset_index(drop=True)
        
        (
            self.sorted_gene_ids,
            self.gene_to_iso_map,
            self.iso_to_gene_index,
            self.isoform_ids,
        ) = build_gene_isoform_mapping(
            self.iso_var,
            gene_col=gene_id_col,
            isoform_col=isoform_id_col if isoform_id_col in self.iso_var.columns else None,
        )
        
        self.n_isoforms = len(self.isoform_ids)
        print(f"Final isoform count: {self.n_isoforms}")
        
        self.iso_to_gene_index_tensor = torch.from_numpy(self.iso_to_gene_index).long()

        # 5. File Handles / In-Memory Load
        self.genes_file = None
        self.iso_file = None
        self.genes_dataset = None
        self.iso_dataset = None
        
        self.genes_data = None
        self.iso_data = None
        
        if self.in_memory:
            self.load_in_memory(genes_backed, isoforms_backed)
            
    def load_in_memory(self, genes_backed, isoforms_backed):
        print("Loading dataset into memory...")
        # We use load_dense_subset which handles sparse reading and slicing
        self.genes_data = load_dense_subset(genes_backed, self.sample_idx, self.gene_mask)
        self.iso_data = load_dense_subset(isoforms_backed, self.sample_idx, self.iso_mask)
        print("Dataset loaded into memory.")

    def _open_files(self):
        if self.in_memory:
            return

        if self.genes_file is None:
            self.genes_file = h5py.File(self.genes_path, "r")
            self.genes_dataset = self.genes_file["X"]
        if self.iso_file is None:
            self.iso_file = h5py.File(self.isoform_path, "r")
            self.iso_dataset = self.iso_file["X"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.in_memory:
            gene_row = self.genes_data[idx]
            iso_row = self.iso_data[idx]
        else:
            if self.genes_file is None:
                self._open_files()
            
            # Map logical index to physical index
            file_idx = self.sample_idx[idx]
            
            def read_filtered_row(dataset, row_i, col_indices):
                col_indices = np.asarray(col_indices)
                if isinstance(dataset, h5py.Group): # Sparse
                    indptr = dataset["indptr"]
                    start = int(indptr[row_i])
                    end = int(indptr[row_i+1])
                    data = dataset["data"][start:end]
                    indices = dataset["indices"][start:end]
                    
                    if data.size == 0:
                        return np.zeros(len(col_indices), dtype=np.float32)
                    
                    # Map sparse indices to positions within the selected columns without expanding the full row
                    pos = np.searchsorted(col_indices, indices)
                    valid = (pos < len(col_indices)) & (col_indices[pos] == indices)
                    
                    row_filtered = np.zeros(len(col_indices), dtype=np.float32)
                    row_filtered[pos[valid]] = data[valid]
                    return row_filtered
                else:
                    # Dense dataset
                    row = dataset[row_i]
                    return row[col_indices]

            gene_row = read_filtered_row(self.genes_dataset, file_idx, self.gene_indices)
            iso_row = read_filtered_row(self.iso_dataset, file_idx, self.iso_indices)
        
        # Processing
        
        # Genes: normalize and log1p
        # Match load_geneaware: sc.pp.normalize_total(target_sum=1e4) -> log1p
        total_counts = gene_row.sum()
        if total_counts > 0:
            gene_row = gene_row / total_counts * 1e4
        gene_row = np.log1p(gene_row)
        
        # Isoforms: normalize and calculate proportions
        # Match load_geneaware: sc.pp.normalize_total(target_sum=1e4)
        iso_total = iso_row.sum()
        if iso_total > 0:
            iso_row = iso_row / iso_total * 1e4
            
        iso_tensor = torch.from_numpy(iso_row).float()
        
        # Calculate proportions
        gene_sums = torch.zeros(self.n_genes, dtype=torch.float32)
        gene_sums.scatter_add_(0, self.iso_to_gene_index_tensor, iso_tensor)
        
        gene_sums_expanded = gene_sums[self.iso_to_gene_index_tensor]
        proportions = iso_tensor / gene_sums_expanded.clamp_min(1e-8)

        return torch.from_numpy(gene_row).float(), proportions

def save_preprocessed_data(
    save_path: Path,
    genes_tensor: torch.Tensor,
    isoform_proportions: torch.Tensor,
    gene_to_iso_map: Dict[str, List[int]],
    iso_to_gene_index: np.ndarray,
    isoform_ids: List[str],
    gene_ids: List[str] | None = None,
) -> None:
    """
    Save preprocessed gene-aware data to avoid reloading and filtering the full dataset.
    Saves as a single .pt file with all necessary data.
    """
    save_dict = {
        "genes_tensor": genes_tensor,
        "isoform_proportions": isoform_proportions,
        "gene_to_iso_map": gene_to_iso_map,
        "iso_to_gene_index": iso_to_gene_index,
        "isoform_ids": isoform_ids,
    }
    if gene_ids is not None:
        save_dict["gene_ids"] = gene_ids
    torch.save(save_dict, save_path)
    print(f"Saved preprocessed data to {save_path}")
    print(f"  Genes: {genes_tensor.shape[1]}, Isoforms: {isoform_proportions.shape[1]}, Samples: {genes_tensor.shape[0]}")

def load_embeddings(path: Path, standardize: bool = False):
    """Load VAE embeddings from a .pt file."""
    if path.suffix == ".pt":
        Z = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported embedding file: {path}")

    if not torch.is_tensor(Z):
        raise ValueError(f"Expected tensor in {path}, got {type(Z)}")

    Z = Z.float()
    stats: dict = {}

    if standardize:
        mean = Z.mean(dim=0, keepdim=True)
        std = Z.std(dim=0, keepdim=True).clamp_min(1e-8)
        Z = (Z - mean) / std
        stats = {"mean": mean, "std": std}

    return Z, stats
