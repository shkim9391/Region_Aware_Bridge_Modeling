from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from pandas.api.types import CategoricalDtype

BASE = Path(__file__).resolve().parent

# Load
br = sc.read_h5ad(str(BASE / "output" / "breast_008um.h5ad"))

# Ensure Background category exists, then fill NA (optional; fine to keep)
s = br.obs["cluster_graphbased"]
if not isinstance(s.dtype, CategoricalDtype):
    s = s.astype("category")
if "Background" not in s.cat.categories:
    s = s.cat.add_categories(["Background"])
br.obs["cluster_graphbased"] = s.fillna("Background")

# Filter: remove zero-count bins
# Works for sparse matrices
br = br[br.X.sum(axis=1).A1 > 0].copy()

# (Recommended) exclude Background from DE
br = br[br.obs["cluster_graphbased"] != "Background"].copy()

# Normalize + log
sc.pp.normalize_total(br, target_sum=1e4)
sc.pp.log1p(br)

# Rank markers
sc.tl.rank_genes_groups(br, groupby="cluster_graphbased", method="wilcoxon")

# Save marker table
markers = sc.get.rank_genes_groups_df(br, group=None)
markers.to_csv(BASE / "output" / "breast_rank_genes_groups.csv", index=False)
print("[OK] wrote output/breast_rank_genes_groups.csv")

# Plot top markers
sc.pl.rank_genes_groups(br, n_genes=10, sharey=False)

mapping = {
    # Luminal / epithelial
    "Cluster 8":  "Luminal_Epithelial",
    "Cluster 9":  "Luminal_Epithelial",
    "Cluster 14": "Luminal_Epithelial",
    "Cluster 16": "Luminal_Epithelial",
    "Cluster 17": "Luminal_Epithelial",
    "Cluster 18": "Luminal_Epithelial",

    # Secretory / mucosal-like epithelial
    "Cluster 1":  "Secretory_Epithelial",
    "Cluster 4":  "Secretory_Epithelial",
    "Cluster 12": "Secretory_Epithelial",

    # Basal / myoepithelial / smooth muscle
    "Cluster 3":  "Myoepithelial",
    "Cluster 6":  "Myoepithelial",

    # Fibroblast / stroma
    "Cluster 2":  "Fibroblast",
    "Cluster 7":  "Fibroblast",
    "Cluster 11": "Fibroblast",
    "Cluster 13": "Fibroblast",
    "Cluster 20": "Fibroblast",

    # Stress / metabolic
    "Cluster 5":  "Stressed_Metabolic",
    "Cluster 10": "Stressed_Metabolic",

    # Immune-like
    "Cluster 15": "Immune_Plasma_like",

    # Progenitor / basal-like
    "Cluster 19": "Basal_Progenitor",

    "Background": "Background"
}

br.obs["cell_type_coarse"] = (
    br.obs["cluster_graphbased"]
      .map(mapping)
      .fillna("Unassigned")
)

br.write("output/breast_008um_coarse_annotated.h5ad")
print("[OK] wrote output/breast_008um_coarse_annotated.h5ad")
