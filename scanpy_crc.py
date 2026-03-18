from pathlib import Path
import scanpy as sc
import pandas as pd
from pandas.api.types import CategoricalDtype

BASE = Path(__file__).resolve().parent

# Load CRC object
crc = sc.read_h5ad(str(BASE / "output" / "crc_008um.h5ad"))

# Ensure Background category exists (optional but keeps consistency)
s = crc.obs["cluster_graphbased"]
if not isinstance(s.dtype, CategoricalDtype):
    s = s.astype("category")
if "Background" not in s.cat.categories:
    s = s.cat.add_categories(["Background"])
crc.obs["cluster_graphbased"] = s.fillna("Background")

import numpy as np

# Drop bins with zero total counts
crc = crc[crc.X.sum(axis=1).A1 > 0].copy()   # if X is sparse
# If this errors, use:
# crc = crc[crc.obs["total_counts"] > 0].copy()  # if you've computed qc metrics

# Quick normalization for marker discovery
sc.pp.normalize_total(crc, target_sum=1e4)
sc.pp.log1p(crc)

# Rank marker genes per cluster
sc.tl.rank_genes_groups(crc, groupby="cluster_graphbased", method="wilcoxon")

# Show top markers
sc.pl.rank_genes_groups(crc, n_genes=10, sharey=False)

from pathlib import Path
import scanpy as sc

BASE = Path(__file__).resolve().parent

markers = sc.get.rank_genes_groups_df(crc, group=None)
markers.to_csv(BASE / "output" / "crc_rank_genes_groups.csv", index=False)
print("[OK] wrote output/crc_rank_genes_groups.csv")

mapping = {
    "Cluster 1":  "Epithelial/Tumor",
    "Cluster 3":  "Epithelial/Tumor",
    "Cluster 5":  "Epithelial/Tumor",

    "Cluster 10": "Secretory_Epithelial",
    "Cluster 13": "Secretory_Epithelial",
    "Cluster 14": "Secretory_Epithelial",
    "Cluster 15": "Secretory_Epithelial",

    "Cluster 2":  "Fibroblast",
    "Cluster 9":  "SmoothMuscle",

    "Cluster 7":  "Endothelial",

    "Cluster 4":  "Plasma",
    "Cluster 6":  "Plasma",

    "Cluster 8":  "Myeloid",
    "Cluster 12": "Myeloid",
    "Cluster 16": "Inflammatory_Myeloid",
    "Cluster 17": "Inflammatory_Myeloid",

    "Cluster 11": "Stressed_Metabolic",

    "Background": "Background"
}

crc.obs["cell_type_coarse"] = (
    crc.obs["cluster_graphbased"]
       .map(mapping)
       .fillna("Unassigned")
)

crc.write("output/crc_008um_coarse_annotated.h5ad")
