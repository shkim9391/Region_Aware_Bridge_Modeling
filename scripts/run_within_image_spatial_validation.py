import argparse
import math
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGION_ORDER = ["UL", "UR", "LL", "LR"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-image spatial validation for single-image CRC/Breast patch datasets."
    )
    parser.add_argument("--crc-csv", required=True, type=Path, help="CRC patch-index CSV.")
    parser.add_argument("--breast-csv", required=True, type=Path, help="Breast patch-index CSV.")
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory for CSVs, figures, and summary.",
    )
    parser.add_argument(
        "--x-col",
        default="x0",
        type=str,
        help="X coordinate column. Default: x0",
    )
    parser.add_argument(
        "--y-col",
        default="y0",
        type=str,
        help="Y coordinate column. Default: y0",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=["epi_like", "fibroblast", "smooth_myoepi", "ECM"],
        help="Patch-level feature columns to analyze.",
    )
    parser.add_argument(
        "--shift-frac",
        type=float,
        default=0.10,
        help="Shift fraction of image span for shifted partition. Default: 0.10",
    )
    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=45.0,
        help="Rotation angle in degrees for rotated partition. Default: 45",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=500,
        help="Number of shuffle-null permutations per image. Default: 500",
    )
    parser.add_argument(
        "--max-points-plot",
        type=int,
        default=40000,
        help="Maximum sampled points per image for scatter figures. Default: 40000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Default: 0",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# I/O and validation
# ---------------------------------------------------------------------


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_patch_csv(
    path: Path,
    dataset_label: str,
    x_col: str,
    y_col: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    # Read only needed columns when possible.
    optional_cols = ["image_path", "dataset", "barcode", "x_px", "y_px"]
    usecols = None

    try:
        header = pd.read_csv(path, nrows=0)
        needed = [x_col, y_col, *feature_cols]
        available = set(header.columns.tolist())
        cols = [c for c in needed + optional_cols if c in available]
        usecols = cols
    except Exception:
        usecols = None

    df = pd.read_csv(path, usecols=usecols)
    require_columns(df, [x_col, y_col, *feature_cols], path.name)

    for col in [x_col, y_col, *feature_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[x_col, y_col, *feature_cols]).copy()
    dropped = before - len(df)
    if len(df) == 0:
        raise ValueError(f"All rows were dropped from {path.name} after NA filtering.")

    df["dataset_label"] = dataset_label
    df["image_id"] = derive_image_id(df, dataset_label)
    df.attrs["rows_dropped_missing"] = dropped
    df.attrs["source_path"] = str(path)
    return df


def derive_image_id(df: pd.DataFrame, fallback: str) -> str:
    if "image_path" in df.columns:
        stems = pd.Series(df["image_path"]).dropna().astype(str).map(lambda x: Path(x).stem).unique().tolist()
        if len(stems) == 1:
            return stems[0]
        if len(stems) > 1:
            return f"{fallback}_multiimage"
    return fallback


def make_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------


def assign_region_codes(
    x: np.ndarray,
    y: np.ndarray,
    scheme: str,
    shift_frac: float,
    rotate_deg: float,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Return:
        region_codes: int array in {0,1,2,3} corresponding to UL, UR, LL, LR
        scheme_info:  dict with thresholds / transform parameters
        x_use, y_use: transformed coordinates used for partitioning
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    cx = float(np.median(x))
    cy = float(np.median(y))

    x_centered = x - cx
    y_centered = y - cy

    if scheme == "original":
        x_use = x_centered
        y_use = y_centered
        x_thr = 0.0
        y_thr = 0.0

    elif scheme == "shifted":
        x_use = x_centered
        y_use = y_centered
        x_span = float(np.max(x) - np.min(x))
        y_span = float(np.max(y) - np.min(y))
        x_thr = shift_frac * x_span
        y_thr = shift_frac * y_span

    elif scheme == "rotated":
        theta = math.radians(rotate_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_use = x_centered * cos_t - y_centered * sin_t
        y_use = x_centered * sin_t + y_centered * cos_t
        x_thr = 0.0
        y_thr = 0.0

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    left = x_use <= x_thr
    upper = y_use <= y_thr

    region_codes = np.empty(x.shape[0], dtype=np.int8)
    region_codes[left & upper] = 0   # UL
    region_codes[~left & upper] = 1  # UR
    region_codes[left & ~upper] = 2  # LL
    region_codes[~left & ~upper] = 3 # LR

    info = {
        "center_x": cx,
        "center_y": cy,
        "x_threshold": float(x_thr),
        "y_threshold": float(y_thr),
        "shift_frac": float(shift_frac),
        "rotate_deg": float(rotate_deg),
    }
    return region_codes, info, x_use, y_use


def region_names_from_codes(codes: np.ndarray) -> np.ndarray:
    return np.array(REGION_ORDER, dtype=object)[codes]


# ---------------------------------------------------------------------
# Region summaries and metrics
# ---------------------------------------------------------------------


def region_means_from_codes(
    X: np.ndarray,
    region_codes: np.ndarray,
    n_regions: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient region means via bincount.

    Returns:
        means:  (n_regions, n_features)
        counts: (n_regions,)
    """
    X = np.asarray(X, dtype=float)
    region_codes = np.asarray(region_codes, dtype=int)

    counts = np.bincount(region_codes, minlength=n_regions).astype(float)
    means = np.full((n_regions, X.shape[1]), np.nan, dtype=float)

    for j in range(X.shape[1]):
        sums = np.bincount(region_codes, weights=X[:, j], minlength=n_regions).astype(float)
        valid = counts > 0
        means[valid, j] = sums[valid] / counts[valid]

    return means, counts.astype(int)


def compute_whole_means(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float).mean(axis=0)


def pairwise_mean_abs_diff(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals)
    vals = vals[valid]
    if vals.size < 2:
        return np.nan

    diffs = []
    for i in range(vals.size):
        for j in range(i + 1, vals.size):
            diffs.append(abs(vals[i] - vals[j]))
    return float(np.mean(diffs)) if diffs else np.nan


def safe_cv(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=0))
    if np.isclose(mean_val, 0.0):
        return np.nan
    return std_val / abs(mean_val)


def heterogeneity_rows_from_means(
    dataset_label: str,
    image_id: str,
    scheme: str,
    feature_cols: Sequence[str],
    region_means: np.ndarray,
    region_counts: np.ndarray,
    whole_means: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for j, feat in enumerate(feature_cols):
        vals = region_means[:, j]
        vals_valid = vals[np.isfinite(vals)]
        if vals_valid.size == 0:
            continue

        rows.append(
            {
                "dataset": dataset_label,
                "image_id": image_id,
                "scheme": scheme,
                "feature": feat,
                "n_regions": int(np.sum(np.isfinite(vals))),
                "region_range": float(np.nanmax(vals) - np.nanmin(vals)),
                "region_std": float(np.nanstd(vals, ddof=0)),
                "region_cv": float(safe_cv(vals)),
                "mean_abs_pairwise_diff": float(pairwise_mean_abs_diff(vals)),
                "mean_abs_deviation_from_global": float(np.nanmean(np.abs(vals - whole_means[j]))),
                "max_abs_deviation_from_global": float(np.nanmax(np.abs(vals - whole_means[j]))),
                "whole_mean": float(whole_means[j]),
                "min_region_mean": float(np.nanmin(vals)),
                "max_region_mean": float(np.nanmax(vals)),
                "min_region_count": int(np.min(region_counts)),
                "max_region_count": int(np.max(region_counts)),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    overall = {
        "dataset": dataset_label,
        "image_id": image_id,
        "scheme": scheme,
        "feature": "__overall__",
        "n_regions": int(df["n_regions"].max()),
        "region_range": float(df["region_range"].mean()),
        "region_std": float(df["region_std"].mean()),
        "region_cv": float(df["region_cv"].dropna().mean()) if df["region_cv"].notna().any() else np.nan,
        "mean_abs_pairwise_diff": float(df["mean_abs_pairwise_diff"].mean()),
        "mean_abs_deviation_from_global": float(df["mean_abs_deviation_from_global"].mean()),
        "max_abs_deviation_from_global": float(df["max_abs_deviation_from_global"].mean()),
        "whole_mean": np.nan,
        "min_region_mean": np.nan,
        "max_region_mean": np.nan,
        "min_region_count": int(df["min_region_count"].min()),
        "max_region_count": int(df["max_region_count"].max()),
    }
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)


def region_means_long_df(
    dataset_label: str,
    image_id: str,
    scheme: str,
    feature_cols: Sequence[str],
    region_means: np.ndarray,
    region_counts: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for region_idx, region_name in enumerate(REGION_ORDER):
        for j, feat in enumerate(feature_cols):
            rows.append(
                {
                    "dataset": dataset_label,
                    "image_id": image_id,
                    "scheme": scheme,
                    "region": region_name,
                    "feature": feat,
                    "mean_value": float(region_means[region_idx, j]),
                    "n_points_region": int(region_counts[region_idx]),
                }
            )
    return pd.DataFrame(rows)


def whole_means_long_df(
    dataset_label: str,
    image_id: str,
    feature_cols: Sequence[str],
    whole_means: np.ndarray,
    n_points: int,
) -> pd.DataFrame:
    rows = []
    for j, feat in enumerate(feature_cols):
        rows.append(
            {
                "dataset": dataset_label,
                "image_id": image_id,
                "feature": feat,
                "whole_mean": float(whole_means[j]),
                "n_points_image": int(n_points),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Shuffle null
# ---------------------------------------------------------------------


def metrics_wide_to_long(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    source: str,
    perm_id: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for metric in metric_cols:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "image_id": row["image_id"],
                    "scheme": row["scheme"],
                    "feature": row["feature"],
                    "source": source,
                    "perm_id": perm_id,
                    "metric_name": metric,
                    "value": row[metric],
                }
            )
    return pd.DataFrame(rows)


def run_region_label_shuffle_null(
    dataset_label: str,
    image_id: str,
    X: np.ndarray,
    observed_region_codes: np.ndarray,
    feature_cols: Sequence[str],
    whole_means: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        raw_long_df     one row per perm x feature x metric
        summary_long_df observed vs null summary
    """
    metric_cols = [
        "region_range",
        "region_std",
        "region_cv",
        "mean_abs_pairwise_diff",
        "mean_abs_deviation_from_global",
        "max_abs_deviation_from_global",
    ]

    obs_means, obs_counts = region_means_from_codes(X, observed_region_codes, n_regions=4)
    obs_wide = heterogeneity_rows_from_means(
        dataset_label=dataset_label,
        image_id=image_id,
        scheme="original",
        feature_cols=feature_cols,
        region_means=obs_means,
        region_counts=obs_counts,
        whole_means=whole_means,
    )
    obs_long = metrics_wide_to_long(obs_wide, metric_cols=metric_cols, source="observed", perm_id=None)

    raw_parts = []
    for perm_id in range(n_perm):
        shuffled = rng.permutation(observed_region_codes)
        perm_means, perm_counts = region_means_from_codes(X, shuffled, n_regions=4)
        perm_wide = heterogeneity_rows_from_means(
            dataset_label=dataset_label,
            image_id=image_id,
            scheme="shuffle_null",
            feature_cols=feature_cols,
            region_means=perm_means,
            region_counts=perm_counts,
            whole_means=whole_means,
        )
        raw_parts.append(
            metrics_wide_to_long(
                perm_wide,
                metric_cols=metric_cols,
                source="null",
                perm_id=perm_id,
            )
        )

    raw_long = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()

    summary_rows = []
    group_keys = ["dataset", "image_id", "feature", "metric_name"]

    if len(raw_long) == 0:
        return raw_long, pd.DataFrame()

    obs_subset = obs_long[group_keys + ["value"]].rename(columns={"value": "observed_value"})
    for _, obs_row in obs_subset.iterrows():
        mask = (
            (raw_long["dataset"] == obs_row["dataset"])
            & (raw_long["image_id"] == obs_row["image_id"])
            & (raw_long["feature"] == obs_row["feature"])
            & (raw_long["metric_name"] == obs_row["metric_name"])
        )
        null_vals = raw_long.loc[mask, "value"].dropna().to_numpy(dtype=float)
        observed_value = float(obs_row["observed_value"])

        if null_vals.size == 0:
            null_mean = np.nan
            null_sd = np.nan
            z_score = np.nan
            p_emp = np.nan
        else:
            null_mean = float(np.mean(null_vals))
            null_sd = float(np.std(null_vals, ddof=0))
            z_score = float((observed_value - null_mean) / null_sd) if not np.isclose(null_sd, 0.0) else np.nan
            # Larger observed = more spatial heterogeneity.
            p_emp = float((1 + np.sum(null_vals >= observed_value)) / (1 + null_vals.size))

        summary_rows.append(
            {
                "dataset": obs_row["dataset"],
                "image_id": obs_row["image_id"],
                "feature": obs_row["feature"],
                "metric_name": obs_row["metric_name"],
                "observed_value": observed_value,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z_score": z_score,
                "empirical_p_upper": p_emp,
                "n_perm": int(null_vals.size),
            }
        )

    summary_long = pd.DataFrame(summary_rows)
    return raw_long, summary_long


# ---------------------------------------------------------------------
# Partition sensitivity
# ---------------------------------------------------------------------


def compute_partition_sensitivity(
    region_long: pd.DataFrame,
    heterogeneity_wide: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        metric_change_df    original vs alternative changes for heterogeneity metrics
        region_corr_df      original vs alternative region-mean Spearman correlations
    """
    selected_metrics = [
        "region_range",
        "region_std",
        "region_cv",
        "mean_abs_pairwise_diff",
        "mean_abs_deviation_from_global",
        "max_abs_deviation_from_global",
    ]

    # Metric changes
    change_rows = []
    for (dataset, image_id, feature), sub in heterogeneity_wide.groupby(["dataset", "image_id", "feature"]):
        if "original" not in sub["scheme"].values:
            continue
        orig = sub[sub["scheme"] == "original"].iloc[0]
        for alt_scheme in ["shifted", "rotated"]:
            alt_sub = sub[sub["scheme"] == alt_scheme]
            if len(alt_sub) == 0:
                continue
            alt = alt_sub.iloc[0]
            for metric in selected_metrics:
                orig_val = orig[metric]
                alt_val = alt[metric]
                if pd.isna(orig_val) or pd.isna(alt_val):
                    pct_change = np.nan
                elif np.isclose(orig_val, 0.0):
                    pct_change = np.nan
                else:
                    pct_change = 100.0 * (alt_val - orig_val) / abs(orig_val)
                change_rows.append(
                    {
                        "dataset": dataset,
                        "image_id": image_id,
                        "feature": feature,
                        "scheme_ref": "original",
                        "scheme_alt": alt_scheme,
                        "metric_name": metric,
                        "original_value": float(orig_val) if pd.notna(orig_val) else np.nan,
                        "alt_value": float(alt_val) if pd.notna(alt_val) else np.nan,
                        "delta": float(alt_val - orig_val) if pd.notna(orig_val) and pd.notna(alt_val) else np.nan,
                        "pct_change": float(pct_change) if pd.notna(pct_change) else np.nan,
                    }
                )

    metric_change_df = pd.DataFrame(change_rows)

    # Region-mean correlations
    corr_rows = []
    for (dataset, image_id, feature), sub in region_long.groupby(["dataset", "image_id", "feature"]):
        pivot = (
            sub.pivot_table(
                index="region",
                columns="scheme",
                values="mean_value",
                aggfunc="first",
            )
            .reindex(REGION_ORDER)
        )

        if "original" not in pivot.columns:
            continue

        orig_vec = pivot["original"]
        for alt_scheme in ["shifted", "rotated"]:
            if alt_scheme not in pivot.columns:
                continue
            alt_vec = pivot[alt_scheme]
            rho = orig_vec.corr(alt_vec, method="spearman")
            mad = float(np.nanmean(np.abs(orig_vec.to_numpy(dtype=float) - alt_vec.to_numpy(dtype=float))))
            corr_rows.append(
                {
                    "dataset": dataset,
                    "image_id": image_id,
                    "feature": feature,
                    "scheme_ref": "original",
                    "scheme_alt": alt_scheme,
                    "spearman_rho_regions": float(rho) if pd.notna(rho) else np.nan,
                    "mean_abs_region_shift": mad,
                }
            )

    region_corr_df = pd.DataFrame(corr_rows)
    return metric_change_df, region_corr_df


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def sample_points_for_plot(df: pd.DataFrame, n_max: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n_max:
        return df.copy()
    idx = rng.choice(np.arange(len(df)), size=n_max, replace=False)
    return df.iloc[idx].copy()


def plot_original_partition_scatter(
    sampled_assignments: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    datasets = sampled_assignments["dataset"].astype(str).unique().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)

    region_colors = {
        "UL": "#1f77b4",
        "UR": "#ff7f0e",
        "LL": "#2ca02c",
        "LR": "#d62728",
    }

    for ax, dataset in zip(axes.flatten(), datasets):
        sub = sampled_assignments[sampled_assignments["dataset"] == dataset].copy()
        for region in REGION_ORDER:
            s = sub[sub["region"] == region]
            ax.scatter(
                s[x_col],
                s[y_col],
                s=2,
                alpha=0.5,
                color=region_colors[region],
                label=region,
            )
        ax.set_title(f"{dataset}: original 2x2 partition")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(frameon=False, markerscale=4)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_original_region_heatmaps(
    region_long: pd.DataFrame,
    feature_cols: Sequence[str],
    out_pdf: Path,
    out_png: Path,
) -> None:
    sub = region_long[region_long["scheme"] == "original"].copy()
    datasets = sub["dataset"].astype(str).unique().tolist()

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes.flatten(), datasets):
        s = sub[sub["dataset"] == dataset].copy()
        mat = (
            s.pivot_table(index="region", columns="feature", values="mean_value", aggfunc="first")
            .reindex(REGION_ORDER)
            .reindex(columns=feature_cols)
        )
        im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(np.arange(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(REGION_ORDER)))
        ax.set_yticklabels(REGION_ORDER)
        ax.set_title(f"{dataset}: regional means")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Mean feature value", rotation=90)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_shuffle_null_histograms(
    shuffle_raw: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    metric_name: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    # Focus on overall metric for readability.
    raw = shuffle_raw[
        (shuffle_raw["feature"] == "__overall__")
        & (shuffle_raw["metric_name"] == metric_name)
    ].copy()

    summary = shuffle_summary[
        (shuffle_summary["feature"] == "__overall__")
        & (shuffle_summary["metric_name"] == metric_name)
    ].copy()

    datasets = raw["dataset"].astype(str).unique().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)

    for ax, dataset in zip(axes.flatten(), datasets):
        r = raw[raw["dataset"] == dataset]
        s = summary[summary["dataset"] == dataset]
        ax.hist(r["value"].dropna().to_numpy(dtype=float), bins=30, alpha=0.75)
        if len(s) > 0:
            obs = float(s["observed_value"].iloc[0])
            p_emp = float(s["empirical_p_upper"].iloc[0])
            ax.axvline(obs, linestyle="--", linewidth=2)
            ax.set_title(f"{dataset}: {metric_name}\nobserved={obs:.4f}, p={p_emp:.4g}")
        else:
            ax.set_title(f"{dataset}: {metric_name}")
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Permutation count")

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_partition_sensitivity(
    heterogeneity_wide: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    sub = heterogeneity_wide[heterogeneity_wide["feature"] == "__overall__"].copy()
    scheme_order = ["original", "shifted", "rotated"]
    metrics = ["region_range", "mean_abs_pairwise_diff"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    for ax, metric in zip(axes.flatten(), metrics):
        for dataset, s in sub.groupby("dataset"):
            s = s.set_index("scheme").reindex(scheme_order).reset_index()
            ax.plot(s["scheme"], s[metric], marker="o", label=dataset)
        ax.set_title(metric)
        ax.set_xlabel("Partition scheme")
        ax.set_ylabel("Value")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_combined_validation_figure(
    sampled_assignments: pd.DataFrame,
    region_long: pd.DataFrame,
    heterogeneity_wide: pd.DataFrame,
    shuffle_raw: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    feature_cols: Sequence[str],
    out_pdf: Path,
    out_png: Path,
) -> None:
    metric_name = "mean_abs_pairwise_diff"
    raw = shuffle_raw[
        (shuffle_raw["feature"] == "__overall__")
        & (shuffle_raw["metric_name"] == metric_name)
    ].copy()
    summ = shuffle_summary[
        (shuffle_summary["feature"] == "__overall__")
        & (shuffle_summary["metric_name"] == metric_name)
    ].copy()

    datasets = ["CRC", "Breast"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: CRC original partition
    ax = axes[0, 0]
    s = sampled_assignments[sampled_assignments["dataset"] == "CRC"].copy()
    region_colors = {"UL": "#1f77b4", "UR": "#ff7f0e", "LL": "#2ca02c", "LR": "#d62728"}
    for region in REGION_ORDER:
        z = s[s["region"] == region]
        ax.scatter(z[x_col], z[y_col], s=2, alpha=0.5, color=region_colors[region], label=region)
    ax.set_title("A. CRC original partition")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False, markerscale=4)

    # Panel B: Breast original partition
    ax = axes[0, 1]
    s = sampled_assignments[sampled_assignments["dataset"] == "Breast"].copy()
    for region in REGION_ORDER:
        z = s[s["region"] == region]
        ax.scatter(z[x_col], z[y_col], s=2, alpha=0.5, color=region_colors[region], label=region)
    ax.set_title("B. Breast original partition")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False, markerscale=4)

    # Panel C: regional mean heatmap for CRC original
    ax = axes[1, 0]
    s = region_long[(region_long["scheme"] == "original") & (region_long["dataset"] == "CRC")].copy()
    mat = (
        s.pivot_table(index="region", columns="feature", values="mean_value", aggfunc="first")
        .reindex(REGION_ORDER)
        .reindex(columns=feature_cols)
    )
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(REGION_ORDER)))
    ax.set_yticklabels(REGION_ORDER)
    ax.set_title("C. CRC regional means")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel D: shuffle-null histograms overlay for overall mean_abs_pairwise_diff
    ax = axes[1, 1]
    for dataset in datasets:
        r = raw[raw["dataset"] == dataset]["value"].dropna().to_numpy(dtype=float)
        if r.size == 0:
            continue
        ax.hist(r, bins=25, alpha=0.45, label=f"{dataset} null")
        srow = summ[summ["dataset"] == dataset]
        if len(srow) > 0:
            obs = float(srow["observed_value"].iloc[0])
            ax.axvline(obs, linestyle="--", linewidth=2, label=f"{dataset} observed")
    ax.set_title("D. Shuffle null: overall mean_abs_pairwise_diff")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Permutation count")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------


def write_run_summary(
    outpath: Path,
    dataset_infos: List[Dict[str, object]],
    heterogeneity_wide: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    partition_changes: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Within-image spatial validation summary\n")
    lines.append("Important caveat: this dataset currently behaves as one image per cohort, so all results below are within-image spatial validation rather than slide-level benchmarking.\n")

    lines.append("Input images:")
    for info in dataset_infos:
        lines.append(
            f"  - {info['dataset']}: image_id={info['image_id']}, n_patches={info['n_patches']}, dropped_missing={info['dropped_missing']}"
        )
    lines.append("")

    # Observed original overall heterogeneity
    lines.append("Observed original-scheme overall heterogeneity:")
    obs = heterogeneity_wide[
        (heterogeneity_wide["scheme"] == "original")
        & (heterogeneity_wide["feature"] == "__overall__")
    ].copy()
    for _, row in obs.iterrows():
        lines.append(
            f"  - {row['dataset']}: mean region_range={row['region_range']:.4f}, "
            f"mean_abs_pairwise_diff={row['mean_abs_pairwise_diff']:.4f}, "
            f"mean_abs_deviation_from_global={row['mean_abs_deviation_from_global']:.4f}"
        )
    lines.append("")

    # Shuffle summary for overall pairwise diff
    lines.append("Shuffle-null summary (overall mean_abs_pairwise_diff):")
    sub = shuffle_summary[
        (shuffle_summary["feature"] == "__overall__")
        & (shuffle_summary["metric_name"] == "mean_abs_pairwise_diff")
    ].copy()
    for _, row in sub.iterrows():
        lines.append(
            f"  - {row['dataset']}: observed={row['observed_value']:.4f}, "
            f"null_mean={row['null_mean']:.4f}, z={row['z_score']:.4f}, "
            f"empirical_p_upper={row['empirical_p_upper']:.4g}"
        )
    lines.append("")

    # Partition sensitivity overall
    lines.append("Partition sensitivity (overall mean_abs_pairwise_diff relative to original):")
    sub = partition_changes[
        (partition_changes["feature"] == "__overall__")
        & (partition_changes["metric_name"] == "mean_abs_pairwise_diff")
    ].copy()
    for _, row in sub.iterrows():
        lines.append(
            f"  - {row['dataset']} {row['scheme_alt']}: "
            f"original={row['original_value']:.4f}, alt={row['alt_value']:.4f}, "
            f"delta={row['delta']:.4f}, pct_change={row['pct_change']:.2f}%"
        )
    lines.append("")

    outpath.write_text("\n".join(lines))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    make_outdir(args.outdir)
    rng = np.random.default_rng(args.seed)

    datasets = [
        ("CRC", args.crc_csv),
        ("Breast", args.breast_csv),
    ]

    dataset_frames: Dict[str, pd.DataFrame] = {}
    dataset_infos: List[Dict[str, object]] = []

    for label, path in datasets:
        df = load_patch_csv(
            path=path,
            dataset_label=label,
            x_col=args.x_col,
            y_col=args.y_col,
            feature_cols=args.feature_cols,
        )
        dataset_frames[label] = df
        dataset_infos.append(
            {
                "dataset": label,
                "image_id": df["image_id"].iloc[0],
                "n_patches": int(len(df)),
                "dropped_missing": int(df.attrs.get("rows_dropped_missing", 0)),
                "source_path": str(path),
            }
        )

    whole_parts = []
    region_parts = []
    heterogeneity_parts = []
    assignment_sample_parts = []
    scheme_info_parts = []

    original_labels_cache: Dict[str, np.ndarray] = {}
    original_whole_means_cache: Dict[str, np.ndarray] = {}
    original_X_cache: Dict[str, np.ndarray] = {}

    for label, df in dataset_frames.items():
        image_id = str(df["image_id"].iloc[0])
        X = df.loc[:, args.feature_cols].to_numpy(dtype=float)
        whole_means = compute_whole_means(X)
        whole_parts.append(
            whole_means_long_df(
                dataset_label=label,
                image_id=image_id,
                feature_cols=args.feature_cols,
                whole_means=whole_means,
                n_points=len(df),
            )
        )

        original_whole_means_cache[label] = whole_means
        original_X_cache[label] = X

        for scheme in ["original", "shifted", "rotated"]:
            region_codes, scheme_info, x_use, y_use = assign_region_codes(
                x=df[args.x_col].to_numpy(dtype=float),
                y=df[args.y_col].to_numpy(dtype=float),
                scheme=scheme,
                shift_frac=args.shift_frac,
                rotate_deg=args.rotate_deg,
            )
            region_means, region_counts = region_means_from_codes(X, region_codes, n_regions=4)

            region_parts.append(
                region_means_long_df(
                    dataset_label=label,
                    image_id=image_id,
                    scheme=scheme,
                    feature_cols=args.feature_cols,
                    region_means=region_means,
                    region_counts=region_counts,
                )
            )

            heterogeneity_parts.append(
                heterogeneity_rows_from_means(
                    dataset_label=label,
                    image_id=image_id,
                    scheme=scheme,
                    feature_cols=args.feature_cols,
                    region_means=region_means,
                    region_counts=region_counts,
                    whole_means=whole_means,
                )
            )

            scheme_info_parts.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": label,
                            "image_id": image_id,
                            "scheme": scheme,
                            **scheme_info,
                            "count_UL": int(region_counts[0]),
                            "count_UR": int(region_counts[1]),
                            "count_LL": int(region_counts[2]),
                            "count_LR": int(region_counts[3]),
                        }
                    ]
                )
            )

            if scheme == "original":
                original_labels_cache[label] = region_codes.copy()

                plot_df = df[[args.x_col, args.y_col]].copy()
                plot_df["dataset"] = label
                plot_df["image_id"] = image_id
                plot_df["region"] = region_names_from_codes(region_codes)
                plot_df = sample_points_for_plot(plot_df, args.max_points_plot, rng)
                assignment_sample_parts.append(plot_df)

    whole_df = pd.concat(whole_parts, ignore_index=True)
    region_df = pd.concat(region_parts, ignore_index=True)
    heterogeneity_df = pd.concat(heterogeneity_parts, ignore_index=True)
    assignment_sample_df = pd.concat(assignment_sample_parts, ignore_index=True)
    scheme_info_df = pd.concat(scheme_info_parts, ignore_index=True)

    whole_df.to_csv(args.outdir / "within_image_whole_means.csv", index=False)
    region_df.to_csv(args.outdir / "within_image_region_means_by_scheme.csv", index=False)
    heterogeneity_df.to_csv(args.outdir / "within_image_heterogeneity_metrics_by_scheme.csv", index=False)
    assignment_sample_df.to_csv(args.outdir / "within_image_original_partition_sampled_points.csv", index=False)
    scheme_info_df.to_csv(args.outdir / "within_image_partition_scheme_info.csv", index=False)

    # Shuffle null
    shuffle_raw_parts = []
    shuffle_summary_parts = []
    for label, df in dataset_frames.items():
        image_id = str(df["image_id"].iloc[0])
        raw_long, summary_long = run_region_label_shuffle_null(
            dataset_label=label,
            image_id=image_id,
            X=original_X_cache[label],
            observed_region_codes=original_labels_cache[label],
            feature_cols=args.feature_cols,
            whole_means=original_whole_means_cache[label],
            n_perm=args.n_perm,
            rng=np.random.default_rng(args.seed + (1 if label == "CRC" else 2)),
        )
        shuffle_raw_parts.append(raw_long)
        shuffle_summary_parts.append(summary_long)

    shuffle_raw_df = pd.concat(shuffle_raw_parts, ignore_index=True)
    shuffle_summary_df = pd.concat(shuffle_summary_parts, ignore_index=True)

    shuffle_raw_df.to_csv(args.outdir / "within_image_shuffle_null_raw_long.csv", index=False)
    shuffle_summary_df.to_csv(args.outdir / "within_image_shuffle_null_summary.csv", index=False)

    # Partition sensitivity
    partition_change_df, region_corr_df = compute_partition_sensitivity(
        region_long=region_df,
        heterogeneity_wide=heterogeneity_df,
    )
    partition_change_df.to_csv(args.outdir / "within_image_partition_metric_changes.csv", index=False)
    region_corr_df.to_csv(args.outdir / "within_image_partition_region_correlations.csv", index=False)

    # Figures
    plot_original_partition_scatter(
        sampled_assignments=assignment_sample_df,
        x_col=args.x_col,
        y_col=args.y_col,
        out_pdf=args.outdir / "figure_original_partition_scatter.pdf",
        out_png=args.outdir / "figure_original_partition_scatter.png",
    )

    plot_original_region_heatmaps(
        region_long=region_df,
        feature_cols=args.feature_cols,
        out_pdf=args.outdir / "figure_original_region_heatmaps.pdf",
        out_png=args.outdir / "figure_original_region_heatmaps.png",
    )

    plot_shuffle_null_histograms(
        shuffle_raw=shuffle_raw_df,
        shuffle_summary=shuffle_summary_df,
        metric_name="mean_abs_pairwise_diff",
        out_pdf=args.outdir / "figure_shuffle_null_mean_abs_pairwise_diff.pdf",
        out_png=args.outdir / "figure_shuffle_null_mean_abs_pairwise_diff.png",
    )

    plot_partition_sensitivity(
        heterogeneity_wide=heterogeneity_df,
        out_pdf=args.outdir / "figure_partition_sensitivity.pdf",
        out_png=args.outdir / "figure_partition_sensitivity.png",
    )

    plot_combined_validation_figure(
        sampled_assignments=assignment_sample_df,
        region_long=region_df,
        heterogeneity_wide=heterogeneity_df,
        shuffle_raw=shuffle_raw_df,
        shuffle_summary=shuffle_summary_df,
        x_col=args.x_col,
        y_col=args.y_col,
        feature_cols=args.feature_cols,
        out_pdf=args.outdir / "figure_within_image_validation_main.pdf",
        out_png=args.outdir / "figure_within_image_validation_main.png",
    )

    # Run summary
    write_run_summary(
        outpath=args.outdir / "within_image_validation_summary.txt",
        dataset_infos=dataset_infos,
        heterogeneity_wide=heterogeneity_df,
        shuffle_summary=shuffle_summary_df,
        partition_changes=partition_change_df,
    )

    # Console summary
    print("\nWithin-image spatial validation complete.\n")
    for info in dataset_infos:
        print(
            f"{info['dataset']}: image_id={info['image_id']}, "
            f"n_patches={info['n_patches']}, dropped_missing={info['dropped_missing']}"
        )

    print("\nObserved original-scheme overall heterogeneity:")
    obs = heterogeneity_df[
        (heterogeneity_df["scheme"] == "original")
        & (heterogeneity_df["feature"] == "__overall__")
    ].copy()
    for _, row in obs.iterrows():
        print(
            f"  {row['dataset']}: "
            f"region_range={row['region_range']:.4f}, "
            f"mean_abs_pairwise_diff={row['mean_abs_pairwise_diff']:.4f}, "
            f"mean_abs_deviation_from_global={row['mean_abs_deviation_from_global']:.4f}"
        )

    print("\nShuffle-null summary for overall mean_abs_pairwise_diff:")
    sub = shuffle_summary_df[
        (shuffle_summary_df["feature"] == "__overall__")
        & (shuffle_summary_df["metric_name"] == "mean_abs_pairwise_diff")
    ].copy()
    for _, row in sub.iterrows():
        print(
            f"  {row['dataset']}: observed={row['observed_value']:.4f}, "
            f"null_mean={row['null_mean']:.4f}, z={row['z_score']:.4f}, "
            f"empirical_p_upper={row['empirical_p_upper']:.4g}"
        )

    print(f"\nOutputs written to:\n{args.outdir}")


if __name__ == "__main__":
    main()
