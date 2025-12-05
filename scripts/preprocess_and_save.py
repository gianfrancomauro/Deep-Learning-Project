from pathlib import Path
from scripts.data import load_geneaware, save_preprocessed_data, load_preprocessed_data

# Configuration - match your training settings if you want to run on your machine
DATA_DIR = Path("/zhome/af/a/221977/data")

# Point to the held-out split
GENE_PATH = DATA_DIR / "sc_processed_genes.h5ad"
ISOFORM_PATH = DATA_DIR / "sc_processed_transcripts.h5ad"

# Align test features to the train preprocessing output
TRAIN_FEATURES_PATH = None #DATA_DIR / "preprocessed_train_val_3000hvg.pt"

OUTPUT_PATH = Path("/dtu/blackhole/19/221977/train_val_split_data/preprocessed_train_3000hvg.pt")

MIN_COUNTS = 20
MAX_GENES = 3000
MAX_ISOFORMS = None
MAX_SAMPLES = None
SEED = 42

fixed_gene_ids = None
fixed_isoform_ids = None
if TRAIN_FEATURES_PATH:
    print(f"Loading train feature IDs from {TRAIN_FEATURES_PATH}...")
    _, _, _, _, fixed_isoform_ids, fixed_gene_ids = load_preprocessed_data(TRAIN_FEATURES_PATH, return_gene_ids=True)

print("Loading and preprocessing full dataset (this will take a few minutes)...")
genes_tensor, isoform_proportions, gene_to_iso_map, iso_to_gene_index, isoform_ids, gene_ids = load_geneaware(
    gene_path=GENE_PATH,
    isoform_path=ISOFORM_PATH,
    min_counts=MIN_COUNTS,
    max_genes=MAX_GENES,
    max_isoforms=MAX_ISOFORMS,
    max_samples=MAX_SAMPLES,
    gene_id_col=None,
    isoform_id_col=None,
    seed=SEED,
    selection_method="hvg",
    fixed_gene_ids=fixed_gene_ids,
    fixed_isoform_ids=fixed_isoform_ids,
    return_gene_ids=True,
)

print("Saving preprocessed data...")
save_preprocessed_data(
    OUTPUT_PATH,
    genes_tensor,
    isoform_proportions,
    gene_to_iso_map,
    iso_to_gene_index,
    isoform_ids,
    gene_ids=gene_ids,
)

print("\nDone! Now you can use load_preprocessed_data() in your training scripts.")
print(f"File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")
