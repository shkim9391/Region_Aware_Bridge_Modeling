from pathlib import Path
import pandas as pd
import scanpy as sc

BASE = Path(__file__).resolve().parent
(BASE / "output").mkdir(exist_ok=True)

# Paths
matrix_h5 = BASE / "Breast" / "outs" / "binned_outputs" / "square_008um" / "filtered_feature_bc_matrix.h5"
clusters_csv = BASE / "Graph-Based_breast_cancer_008um.csv"
out_h5ad = BASE / "output" / "breast_008um.h5ad"

# Load matrix
adata = sc.read_10x_h5(str(matrix_h5))
adata.var_names_make_unique()

# Load clusters CSV
br_clusters = pd.read_csv(str(clusters_csv))

# Normalize barcodes
br_clusters["Barcode"] = br_clusters["Barcode"].astype(str).str.replace("-1", "", regex=False)
adata.obs_names = adata.obs_names.astype(str).str.replace("-1", "", regex=False)

# Find cluster column (Graph-based / Graph-Based)
cluster_col = [c for c in br_clusters.columns if "graph" in c.lower()][0]

# Join
adata.obs["cluster_graphbased"] = br_clusters.set_index("Barcode")[cluster_col].reindex(adata.obs_names)

# Save
adata.write(str(out_h5ad))
print(f"[OK] Wrote: {out_h5ad}")
