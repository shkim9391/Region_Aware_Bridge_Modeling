from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_ROOT = PROJECT_ROOT / "data_raw" / "external_solid_tumors"
OUT_ROOT = PROJECT_ROOT / "outputs" / "external_solid_tumors"

PROGRAM_OUT = OUT_ROOT / "01_program_scores"
REGION_OUT = OUT_ROOT / "02_region_summaries"
VALIDATION_OUT = OUT_ROOT / "03_validation"
FIG_OUT = OUT_ROOT / "figures"
SUPP_OUT = OUT_ROOT / "supplementary_tables"

for p in [PROGRAM_OUT, REGION_OUT, VALIDATION_OUT, FIG_OUT, SUPP_OUT]:
    p.mkdir(parents=True, exist_ok=True)


DATASETS = [
    {
        "dataset_id": "lung_cancer",
        "dataset_label": "Lung cancer",
        "disease": "lung_cancer",
        "raw_dir": EXTERNAL_ROOT / "lung_cancer",
    },
    {
        "dataset_id": "prostate_cancer",
        "dataset_label": "Prostate cancer",
        "disease": "prostate_cancer",
        "raw_dir": EXTERNAL_ROOT / "prostate_cancer",
    },
    {
        "dataset_id": "ovarian_cancer",
        "dataset_label": "Ovarian cancer",
        "disease": "ovarian_cancer",
        "raw_dir": EXTERNAL_ROOT / "ovarian_cancer",
    },
]

PROGRAM_GENESETS: Dict[str, List[str]] = {
    "epithelial_like": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1"],
    "fibroblast_stromal": ["DCN", "LUM", "PDGFRA", "PDGFRB", "FAP", "COL1A1", "COL1A2"],
    "smooth_contractile": ["ACTA2", "TAGLN", "MYL9", "MYH11", "CNN1", "TPM2"],
    "ECM": ["COL1A1", "COL1A2", "COL3A1", "FN1", "LUM", "DCN"],
}

PROGRAMS = list(PROGRAM_GENESETS.keys())
REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]
REGION_LABELS = {
    "Q1_UL": "upper_left",
    "Q2_UR": "upper_right",
    "Q3_LL": "lower_left",
    "Q4_LR": "lower_right",
}
METRICS = [
    "region_range",
    "region_std",
    "region_cv",
    "mean_abs_pairwise_diff",
    "mean_abs_deviation_from_global",
    "max_abs_deviation_from_global",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift-frac", type=float, default=0.10)
    ap.add_argument("--rotate-deg", type=float, default=45.0)
    ap.add_argument("--max-points-plot", type=int, default=30000)
    return ap.parse_args()


def find_one(raw_dir: Path, pattern: str) -> Path:
    hits = sorted(raw_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {raw_dir}")
    return hits[0]


def matrix_max(x) -> float:
    xmax = x.max()
    if hasattr(xmax, "A1"):
        return float(xmax.A1[0])
    return float(xmax)


def normalize_if_needed(adata: sc.AnnData, threshold: float = 50.0) -> str:
    xmax = matrix_max(adata.X)
    if xmax <= threshold:
        return f"skipped_assumed_processed_xmax_{xmax:.3f}"
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return f"normalize_total_1e4_log1p_xmax_before_{xmax:.3f}"


def read_positions(path: Path) -> pd.DataFrame:
    pos = pd.read_parquet(path)

    # Normalize likely Space Ranger column names.
    rename = {}
    for c in pos.columns:
        lc = c.lower()
        if lc in ["barcode", "barcodes"]:
            rename[c] = "barcode"
        elif lc in ["pxl_col_in_fullres", "pixel_col", "x", "x0"]:
            rename[c] = "x0"
        elif lc in ["pxl_row_in_fullres", "pixel_row", "y", "y0"]:
            rename[c] = "y0"
        elif lc == "in_tissue":
            rename[c] = "in_tissue"
        elif lc == "array_row":
            rename[c] = "array_row"
        elif lc == "array_col":
            rename[c] = "array_col"

    pos = pos.rename(columns=rename)

    required = ["barcode", "x0", "y0"]
    missing = [c for c in required if c not in pos.columns]
    if missing:
        raise ValueError(f"Positions file missing {missing}; columns={list(pos.columns)}")

    if "in_tissue" in pos.columns:
        pos = pos.loc[pd.to_numeric(pos["in_tissue"], errors="coerce") == 1].copy()

    pos["barcode"] = pos["barcode"].astype(str)
    pos["x0"] = pd.to_numeric(pos["x0"], errors="coerce")
    pos["y0"] = pd.to_numeric(pos["y0"], errors="coerce")
    pos = pos.dropna(subset=["barcode", "x0", "y0"]).copy()

    return pos


def score_programs(adata: sc.AnnData) -> Dict[str, str]:
    status = {}
    for program, genes in PROGRAM_GENESETS.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 2:
            adata.obs[program] = 0.0
            status[program] = f"set_zero_present_{len(present)}"
        else:
            sc.tl.score_genes(adata, gene_list=present, score_name=program, use_raw=False)
            status[program] = f"scored_present_{len(present)}"
    return status


def load_and_score_dataset(ds: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    dataset_id = str(ds["dataset_id"])
    dataset_label = str(ds["dataset_label"])
    raw_dir = Path(ds["raw_dir"])

    h5_path = find_one(raw_dir, "filtered_feature_bc_matrix.h5")
    pos_path = find_one(raw_dir, "tissue_positions.parquet")

    print(f"\nLoading {dataset_label}")
    print(f"  matrix: {h5_path}")
    print(f"  positions: {pos_path}")

    adata = sc.read_10x_h5(str(h5_path), gex_only=True)
    adata.var_names_make_unique()

    n_matrix = adata.n_obs

    pos = read_positions(pos_path)
    n_positions = len(pos)

    common = adata.obs_names.intersection(pd.Index(pos["barcode"].astype(str)))
    if len(common) == 0:
        raise ValueError(f"No barcode overlap between matrix and positions for {dataset_label}")

    adata = adata[common].copy()
    pos = pos.set_index("barcode").loc[adata.obs_names].reset_index()

    # Remove zero-count bins before normalization and program scoring.
    row_sums = adata.X.sum(axis=1)
    if hasattr(row_sums, "A1"):
        row_sums = row_sums.A1
    else:
        row_sums = np.asarray(row_sums).ravel()

    nonzero_mask = row_sums > 0
    n_before_zero_filter = int(adata.n_obs)
    n_zero_removed = int(np.sum(~nonzero_mask))

    adata = adata[nonzero_mask].copy()
    pos = pos.loc[nonzero_mask].reset_index(drop=True)

    normalization_status = normalize_if_needed(adata)
    program_status = score_programs(adata)

    out = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "dataset_label": dataset_label,
            "disease": ds["disease"],
            "barcode": adata.obs_names.astype(str),
            "x0": pos["x0"].to_numpy(dtype=float),
            "y0": pos["y0"].to_numpy(dtype=float),
        }
    )

    for col in ["array_row", "array_col"]:
        if col in pos.columns:
            out[col] = pos[col].values

    for program in PROGRAMS:
        out[program] = adata.obs[program].to_numpy(dtype=float)

    score_out = PROGRAM_OUT / f"{dataset_id}_external_program_scores_core4.csv"
    out.to_csv(score_out, index=False)

    summary = {
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "disease": ds["disease"],
        "matrix_file": str(h5_path),
        "positions_file": str(pos_path),
        "n_matrix_barcodes": int(n_matrix),
        "n_tissue_positions": int(n_positions),
        "n_before_zero_filter": int(n_before_zero_filter),
        "n_zero_count_removed": int(n_zero_removed),
        "n_intersection_used": int(len(out)),
        "normalization_status": normalization_status,
        "program_score_csv": str(score_out),
    }
    for program, status in program_status.items():
        summary[f"{program}_status"] = status

    print(f"  [OK] wrote {score_out}")
    print(f"  n used: {len(out)}")
    print(f"  normalization: {normalization_status}")

    return out, summary


def assign_partition(x: np.ndarray, y: np.ndarray, scheme: str, shift_frac: float, rotate_deg: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_med = float(np.median(x))
    y_med = float(np.median(y))

    if scheme == "original":
        x_use, y_use = x, y
        x_cut, y_cut = x_med, y_med
    elif scheme == "shifted":
        x_use, y_use = x, y
        x_cut = x_med + shift_frac * float(np.max(x) - np.min(x))
        y_cut = y_med + shift_frac * float(np.max(y) - np.min(y))
    elif scheme == "rotated":
        theta = math.radians(rotate_deg)
        x_c = x - x_med
        y_c = y - y_med
        x_use = x_c * math.cos(theta) - y_c * math.sin(theta)
        y_use = x_c * math.sin(theta) + y_c * math.cos(theta)
        x_cut, y_cut = 0.0, 0.0
    else:
        raise ValueError(scheme)

    left = x_use <= x_cut
    upper = y_use <= y_cut
    codes = np.empty(len(x), dtype=np.int8)
    codes[left & upper] = 0
    codes[(~left) & upper] = 1
    codes[left & (~upper)] = 2
    codes[(~left) & (~upper)] = 3
    return codes


def region_means(X: np.ndarray, codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(codes, minlength=4).astype(float)
    means = np.full((4, X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        sums = np.bincount(codes, weights=X[:, j], minlength=4).astype(float)
        valid = counts > 0
        means[valid, j] = sums[valid] / counts[valid]
    return means, counts.astype(int)


def safe_cv(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    return np.nan if np.isclose(mu, 0.0) else sd / abs(mu)


def mean_abs_pairwise(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    diffs = []
    for i in range(vals.size):
        for j in range(i + 1, vals.size):
            diffs.append(abs(vals[i] - vals[j]))
    return float(np.mean(diffs))


def heterogeneity(dataset_id: str, dataset_label: str, scheme: str, means: np.ndarray, counts: np.ndarray, whole: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, program in enumerate(PROGRAMS):
        vals = means[:, j]
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "scheme": scheme,
                "program": program,
                "region_range": float(np.nanmax(vals) - np.nanmin(vals)),
                "region_std": float(np.nanstd(vals, ddof=0)),
                "region_cv": float(safe_cv(vals)),
                "mean_abs_pairwise_diff": float(mean_abs_pairwise(vals)),
                "mean_abs_deviation_from_global": float(np.nanmean(np.abs(vals - whole[j]))),
                "max_abs_deviation_from_global": float(np.nanmax(np.abs(vals - whole[j]))),
                "whole_mean": float(whole[j]),
                "min_region_count": int(np.min(counts)),
                "max_region_count": int(np.max(counts)),
            }
        )

    df = pd.DataFrame(rows)
    overall = {
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "scheme": scheme,
        "program": "__overall__",
        "region_range": float(df["region_range"].mean()),
        "region_std": float(df["region_std"].mean()),
        "region_cv": float(df["region_cv"].dropna().mean()) if df["region_cv"].notna().any() else np.nan,
        "mean_abs_pairwise_diff": float(df["mean_abs_pairwise_diff"].mean()),
        "mean_abs_deviation_from_global": float(df["mean_abs_deviation_from_global"].mean()),
        "max_abs_deviation_from_global": float(df["max_abs_deviation_from_global"].mean()),
        "whole_mean": np.nan,
        "min_region_count": int(df["min_region_count"].min()),
        "max_region_count": int(df["max_region_count"].max()),
    }
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)


def region_long(dataset_id: str, dataset_label: str, scheme: str, means: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    rows = []
    for ridx, rid in enumerate(REGION_ORDER):
        for j, program in enumerate(PROGRAMS):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "scheme": scheme,
                    "region_id": rid,
                    "region_label": REGION_LABELS[rid],
                    "program": program,
                    "mean_value": float(means[ridx, j]),
                    "n_bins_region": int(counts[ridx]),
                }
            )
    return pd.DataFrame(rows)


def metrics_to_long(df: pd.DataFrame, source: str, perm_id: int | None = None) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for metric in METRICS:
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_label": row["dataset_label"],
                    "scheme": row["scheme"],
                    "program": row["program"],
                    "source": source,
                    "perm_id": perm_id,
                    "metric_name": metric,
                    "value": row[metric],
                }
            )
    return pd.DataFrame(rows)


def shuffle_null(dataset_id: str, dataset_label: str, X: np.ndarray, codes: np.ndarray, whole: np.ndarray, n_perm: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    obs_means, obs_counts = region_means(X, codes)
    obs_het = heterogeneity(dataset_id, dataset_label, "original", obs_means, obs_counts, whole)
    obs_long = metrics_to_long(obs_het, "observed")

    raw_parts = []
    for p in range(n_perm):
        shuf = rng.permutation(codes)
        m, c = region_means(X, shuf)
        h = heterogeneity(dataset_id, dataset_label, "shuffle_null", m, c, whole)
        raw_parts.append(metrics_to_long(h, "null", p))

    raw = pd.concat(raw_parts, ignore_index=True)

    rows = []
    keys = ["dataset_id", "dataset_label", "program", "metric_name"]
    obs = obs_long[keys + ["value"]].rename(columns={"value": "observed_value"})

    for _, r in obs.iterrows():
        mask = (
            (raw["dataset_id"] == r["dataset_id"])
            & (raw["program"] == r["program"])
            & (raw["metric_name"] == r["metric_name"])
        )
        vals = raw.loc[mask, "value"].dropna().to_numpy(float)
        observed = float(r["observed_value"])
        null_mean = float(np.mean(vals))
        null_sd = float(np.std(vals, ddof=0))
        z = float((observed - null_mean) / null_sd) if not np.isclose(null_sd, 0.0) else np.nan
        p_upper = float((1 + np.sum(vals >= observed)) / (1 + len(vals)))

        rows.append(
            {
                **{k: r[k] for k in keys},
                "observed_value": observed,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z_score": z,
                "empirical_p_upper": p_upper,
                "n_perm": int(len(vals)),
            }
        )

    return raw, pd.DataFrame(rows)


def partition_sensitivity(region_df: pd.DataFrame, het_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    changes = []
    for (dataset_id, program), sub in het_df.groupby(["dataset_id", "program"]):
        orig = sub[sub["scheme"] == "original"]
        if orig.empty:
            continue
        orig = orig.iloc[0]
        for alt_scheme in ["shifted", "rotated"]:
            alt = sub[sub["scheme"] == alt_scheme]
            if alt.empty:
                continue
            alt = alt.iloc[0]
            for metric in METRICS:
                ov = float(orig[metric])
                av = float(alt[metric])
                delta = av - ov
                pct = 100 * delta / abs(ov) if not np.isclose(ov, 0.0) else np.nan
                changes.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_label": orig["dataset_label"],
                        "program": program,
                        "scheme_alt": alt_scheme,
                        "metric_name": metric,
                        "original_value": ov,
                        "alt_value": av,
                        "delta": delta,
                        "pct_change": pct,
                    }
                )

    corrs = []
    for (dataset_id, program), sub in region_df.groupby(["dataset_id", "program"]):
        piv = (
            sub.pivot_table(index="region_id", columns="scheme", values="mean_value", aggfunc="first")
            .reindex(REGION_ORDER)
        )
        if "original" not in piv.columns:
            continue
        for alt_scheme in ["shifted", "rotated"]:
            if alt_scheme not in piv.columns:
                continue
            rho = piv["original"].corr(piv[alt_scheme], method="spearman")
            mad = float(np.nanmean(np.abs(piv["original"].to_numpy(float) - piv[alt_scheme].to_numpy(float))))
            label = sub["dataset_label"].iloc[0]
            corrs.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": label,
                    "program": program,
                    "scheme_alt": alt_scheme,
                    "spearman_rho_regions": float(rho) if pd.notna(rho) else np.nan,
                    "mean_abs_region_shift": mad,
                }
            )

    return pd.DataFrame(changes), pd.DataFrame(corrs)


def read_metrics_summary(ds: Dict[str, object]) -> pd.DataFrame:
    raw_dir = Path(ds["raw_dir"])
    hits = sorted(raw_dir.glob("*metrics*summary*.csv"))
    if not hits:
        return pd.DataFrame()
    try:
        df = pd.read_csv(hits[0], encoding="utf-8")
    except Exception:
        df = pd.read_csv(hits[0], encoding="latin1")
    df.insert(0, "dataset_id", ds["dataset_id"])
    df.insert(1, "dataset_label", ds["dataset_label"])
    df.insert(2, "metrics_file", str(hits[0]))
    return df


def plot_heatmap(region_df: pd.DataFrame) -> None:
    sub = region_df[region_df["scheme"] == "original"].copy()
    datasets = sub["dataset_id"].unique().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.3), squeeze=False)

    for ax, did in zip(axes.flatten(), datasets):
        g = sub[sub["dataset_id"] == did]
        label = g["dataset_label"].iloc[0]
        mat = (
            g.pivot_table(index="region_id", columns="program", values="mean_value", aggfunc="first")
            .reindex(REGION_ORDER)
            .reindex(columns=PROGRAMS)
        )
        im = ax.imshow(mat.to_numpy(float), aspect="auto")
        ax.set_title(label)
        ax.set_xticks(np.arange(len(PROGRAMS)))
        ax.set_xticklabels(PROGRAMS, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(REGION_ORDER)))
        ax.set_yticklabels(REGION_ORDER)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "external_solid_tumor_region_program_heatmaps.png", dpi=300)
    fig.savefig(FIG_OUT / "external_solid_tumor_region_program_heatmaps.pdf")
    plt.close(fig)


def plot_validation_summary(shuffle_summary: pd.DataFrame) -> None:
    sub = shuffle_summary[
        (shuffle_summary["program"] == "__overall__")
        & (shuffle_summary["metric_name"] == "mean_abs_pairwise_diff")
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sub["dataset_label"], sub["observed_value"], label="Observed")
    ax.scatter(sub["dataset_label"], sub["null_mean"], marker="o", label="Shuffle null mean")
    ax.set_ylabel("Overall mean absolute pairwise regional difference")
    ax.set_title("External solid-tumor supplementary validation")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "external_solid_tumor_shuffle_summary.png", dpi=300)
    fig.savefig(FIG_OUT / "external_solid_tumor_shuffle_summary.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    all_scores = []
    score_summaries = []
    region_parts = []
    het_parts = []
    shuffle_raw_parts = []
    shuffle_summary_parts = []
    metrics_parts = []

    for idx, ds in enumerate(DATASETS):
        scores, summary = load_and_score_dataset(ds)
        all_scores.append(scores)
        score_summaries.append(summary)

        metrics = read_metrics_summary(ds)
        if not metrics.empty:
            metrics_parts.append(metrics)

        X = scores[PROGRAMS].to_numpy(float)
        whole = X.mean(axis=0)

        original_codes = None

        for scheme in ["original", "shifted", "rotated"]:
            codes = assign_partition(
                scores["x0"].to_numpy(float),
                scores["y0"].to_numpy(float),
                scheme=scheme,
                shift_frac=args.shift_frac,
                rotate_deg=args.rotate_deg,
            )
            if scheme == "original":
                original_codes = codes

            m, c = region_means(X, codes)
            region_parts.append(
                region_long(ds["dataset_id"], ds["dataset_label"], scheme, m, c)
            )
            het_parts.append(
                heterogeneity(ds["dataset_id"], ds["dataset_label"], scheme, m, c, whole)
            )

        raw, summ = shuffle_null(
            dataset_id=ds["dataset_id"],
            dataset_label=ds["dataset_label"],
            X=X,
            codes=original_codes,
            whole=whole,
            n_perm=args.n_perm,
            seed=args.seed + idx + 1,
        )
        shuffle_raw_parts.append(raw)
        shuffle_summary_parts.append(summ)

        print(f"  [OK] {ds['dataset_label']}: shuffle null n_perm={args.n_perm}")

    score_summary_df = pd.DataFrame(score_summaries)
    region_df = pd.concat(region_parts, ignore_index=True)
    het_df = pd.concat(het_parts, ignore_index=True)
    shuffle_raw_df = pd.concat(shuffle_raw_parts, ignore_index=True)
    shuffle_summary_df = pd.concat(shuffle_summary_parts, ignore_index=True)
    part_change_df, part_corr_df = partition_sensitivity(region_df, het_df)

    score_summary_df.to_csv(SUPP_OUT / "external_program_score_run_summary.csv", index=False)
    region_df.to_csv(REGION_OUT / "external_region_program_means_by_scheme.csv", index=False)
    het_df.to_csv(VALIDATION_OUT / "external_heterogeneity_metrics_by_scheme.csv", index=False)
    shuffle_raw_df.to_csv(VALIDATION_OUT / "external_shuffle_null_raw_long.csv", index=False)
    shuffle_summary_df.to_csv(VALIDATION_OUT / "external_shuffle_null_summary.csv", index=False)
    part_change_df.to_csv(VALIDATION_OUT / "external_partition_metric_changes.csv", index=False)
    part_corr_df.to_csv(VALIDATION_OUT / "external_partition_region_correlations.csv", index=False)

    pd.DataFrame(
        [
            {"program": p, "genes": ";".join(g), "n_genes": len(g)}
            for p, g in PROGRAM_GENESETS.items()
        ]
    ).to_csv(SUPP_OUT / "external_program_gene_sets.csv", index=False)

    if metrics_parts:
        pd.concat(metrics_parts, ignore_index=True).to_csv(
            SUPP_OUT / "external_metrics_summary_combined.csv",
            index=False,
        )

    plot_heatmap(region_df)
    plot_validation_summary(shuffle_summary_df)

    overall = shuffle_summary_df[
        (shuffle_summary_df["program"] == "__overall__")
        & (shuffle_summary_df["metric_name"] == "mean_abs_pairwise_diff")
    ][
        [
            "dataset_label",
            "observed_value",
            "null_mean",
            "z_score",
            "empirical_p_upper",
            "n_perm",
        ]
    ]

    print("\nExternal validation summary:")
    print(overall.to_string(index=False))
    print(f"\n[OK] outputs written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
