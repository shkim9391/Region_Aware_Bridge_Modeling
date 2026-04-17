from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

Z_DIMS = [
    "epi_like","fibroblast","endothelial","myeloid","lymphoid_plasma","smooth_myoepi","stressed",
    "IFNG","IFN1","CYTOTOX","AP_MHC","NFKB","PROLIF","HYPOXIA","EMT","OXPHOS","UPR","ECM","ANGIO",
    "log_total_counts","log_n_genes"
]

GENESETS = {
    "IFNG":     ["IFNG","CXCL9","CXCL10","STAT1","IRF1"],
    "IFN1":     ["IFIT1","IFIT3","ISG15","MX1","OAS1"],
    "CYTOTOX":  ["NKG7","PRF1","GZMB","GNLY","CTSW"],
    "AP_MHC":   ["HLA-A","HLA-B","HLA-C","HLA-DRA","HLA-DRB1"],
    "NFKB":     ["NFKBIA","TNFAIP3","REL","ICAM1","CXCL8"],
    "PROLIF":   ["MKI67","TOP2A","HMGB2","CENPF","TYMS"],
    "HYPOXIA":  ["CA9","VEGFA","SLC2A1","LDHA","ALDOA"],
    "EMT":      ["VIM","FN1","SNAI1","ZEB1","ITGA5"],
    "OXPHOS":   ["NDUFA1","NDUFB8","COX6C","ATP5F1A","UQCRC1"],
    "UPR":      ["XBP1","HSPA5","DDIT3","ATF4","DNAJB9"],
    "ECM":      ["COL1A1","COL1A2","COL3A1","LUM","DCN"],
    "ANGIO":    ["PECAM1","VWF","KDR","ESAM","RAMP2"],
}

def add_qc(adata):
    # total counts and number of genes per bin
    if "total_counts" not in adata.obs:
        adata.obs["total_counts"] = np.ravel(adata.X.sum(axis=1))
    if "n_genes" not in adata.obs:
        adata.obs["n_genes"] = np.ravel((adata.X > 0).sum(axis=1))
    adata.obs["log_total_counts"] = np.log1p(adata.obs["total_counts"])
    adata.obs["log_n_genes"] = np.log1p(adata.obs["n_genes"])

def maybe_normalize_log1p(ad):
    """Normalize+log1p only if data doesn't already look log-transformed."""
    x = ad.X
    try:
        xmax = x.max()
        xmax = float(xmax.A1[0]) if hasattr(xmax, "A1") else float(xmax)
    except Exception:
        xmax = float(np.max(x.toarray()))

    # Heuristic threshold
    if xmax <= 50:
        return  # assume already normalized/logged

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

def score_programs(adata):
    # assumes adata is log1p normalized
    for name, genes in GENESETS.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 2:
            adata.obs[name] = 0.0
        else:
            sc.tl.score_genes(adata, gene_list=present, score_name=name, use_raw=False)

def composition_from_cell_type(adata, dataset_name):
    comp = pd.DataFrame(0.0, index=adata.obs_names, columns=[
        "epi_like","fibroblast","endothelial","myeloid","lymphoid_plasma","smooth_myoepi","stressed"
    ])
    ct = adata.obs["cell_type_coarse"].astype(str)

    if dataset_name.lower().startswith("crc"):
        comp.loc[ct.isin(["Epithelial/Tumor","Secretory_Epithelial"]), "epi_like"] = 1.0
        comp.loc[ct.isin(["Fibroblast"]), "fibroblast"] = 1.0
        comp.loc[ct.isin(["Endothelial"]), "endothelial"] = 1.0
        comp.loc[ct.isin(["Myeloid","Inflammatory_Myeloid"]), "myeloid"] = 1.0
        comp.loc[ct.isin(["Plasma"]), "lymphoid_plasma"] = 1.0
        comp.loc[ct.isin(["SmoothMuscle"]), "smooth_myoepi"] = 1.0
        comp.loc[ct.isin(["Stressed_Metabolic"]), "stressed"] = 1.0
    else:
        comp.loc[ct.isin(["Luminal_Epithelial","Secretory_Epithelial","Basal_Progenitor"]), "epi_like"] = 1.0
        comp.loc[ct.isin(["Fibroblast"]), "fibroblast"] = 1.0
        comp.loc[ct.isin(["Myoepithelial"]), "smooth_myoepi"] = 1.0
        comp.loc[ct.isin(["Immune_Plasma_like"]), "lymphoid_plasma"] = 1.0
        comp.loc[ct.isin(["Stressed_Metabolic"]), "stressed"] = 1.0

    return comp

def build_z(dataset_tag, in_h5ad, out_stub):
    ad = sc.read_h5ad(str(in_h5ad))

    # Filter zero-count bins
    ad = ad[ad.X.sum(axis=1).A1 > 0].copy()

    # QC + (maybe) normalize/log
    add_qc(ad)
    maybe_normalize_log1p(ad)

    # Programs
    score_programs(ad)

    # Composition
    comp = composition_from_cell_type(ad, dataset_tag)

    # Assemble Z
    z = pd.DataFrame(index=ad.obs_names)
    for c in comp.columns:
        z[c] = comp[c].values
    for p in ["IFNG","IFN1","CYTOTOX","AP_MHC","NFKB","PROLIF","HYPOXIA","EMT","OXPHOS","UPR","ECM","ANGIO"]:
        z[p] = ad.obs[p].values
    z["log_total_counts"] = ad.obs["log_total_counts"].values
    z["log_n_genes"] = ad.obs["log_n_genes"].values

    z = z[Z_DIMS]
    out_csv = str(out_stub) + ".csv"
    z.reset_index(names="barcode").to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

def main():
    build_z("CRC", OUT / "crc_008um_coarse_annotated.h5ad", OUT / "crc_z_spatial")
    build_z("Breast", OUT / "breast_008um_coarse_annotated.h5ad", OUT / "breast_z_spatial")

if __name__ == "__main__":
    main()
