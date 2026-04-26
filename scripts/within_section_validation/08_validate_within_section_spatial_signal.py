from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "CRC": PROJECT_ROOT / "outputs" / "02_patch_indices" / "crc_patch_index_core4.csv",
    "Breast": PROJECT_ROOT / "outputs" / "02_patch_indices" / "breast_patch_index_core4.csv",
}

OUTDIR_DEFAULT = PROJECT_ROOT / "outputs" / "05_validation"

FEATURE_COLS = [
    "epi_like",
    "fibroblast",
    "smooth_myoepi",
    "ECM",
]

REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]

REGION_LABELS = {
    "Q1_UL": "upper_left",
    "Q2_UR": "upper_right",
    "Q3_LL": "lower_left",
    "Q4_LR": "lower_right",
}

METRIC_COLS = [
    "region_range",
    "region_std",
    "region_cv",
    "mean_abs_pairwise_diff",
    "mean_abs_deviation_from_global",
    "max_abs_deviation_from_global",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-section spatial validation for region-aware bridge summaries."
    )
    parser.add_argument(
        "--crc-csv",
        type=Path,
        default=DEFAULT_INPUTS["CRC"],
        help="Clean CRC core-four patch-index CSV.",
    )
    parser.add_argument(
        "--breast-csv",
        type=Path,
        default=DEFAULT_INPUTS["Breast"],
        help="Clean breast core-four patch-index CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR_DEFAULT,
        help="Output directory.",
    )
    parser.add_argument("--x-col", default="x0", help="X coordinate column.")
    parser.add_argument("--y-col", default="y0", help="Y coordinate column.")
    parser.add_argument(
        "--n-perm",
        type=int,
        default=500,
        help="Number of shuffle-null permutations. Use 500 for test runs and 5000 for final revision outputs.",
    )
    parser.add_argument(
        "--shift-frac",
        type=float,
        default=0.10,
        help="Shift fraction of coordinate span for shifted 2x2 partition.",
    )
    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=45.0,
        help="Rotation angle in degrees for rotated 2x2 partition.",
    )
    parser.add_argument(
        "--max-points-plot",
        type=int,
        default=40000,
        help="Maximum points sampled per section for scatter plots.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_core4_csv(path: Path, dataset_label: str, x_col: str, y_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV for {dataset_label}: {path}")

    needed = [
        "dataset",
        "sample_id",
        "barcode",
        x_col,
        y_col,
        *FEATURE_COLS,
    ]

    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in needed if c in header.columns]
    df = pd.read_csv(path, usecols=usecols)

    require_columns(df, needed, dataset_label)

    df["dataset"] = dataset_label

    for col in [x_col, y_col, *FEATURE_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=[x_col, y_col, *FEATURE_COLS]).copy()
    n_after = len(df)

    df.attrs["n_before"] = n_before
    df.attrs["n_after"] = n_after
    df.attrs["n_removed_missing"] = n_before - n_after

    return df


def assign_partition(
    x: np.ndarray,
    y: np.ndarray,
    scheme: str,
    shift_frac: float,
    rotate_deg: float,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Assign region codes using median-based partitions.

    Region code order:
      0: Q1_UL
      1: Q2_UR
      2: Q3_LL
      3: Q4_LR
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_med = float(np.median(x))
    y_med = float(np.median(y))

    if scheme == "original":
        x_use = x.copy()
        y_use = y.copy()
        x_cut = x_med
        y_cut = y_med

    elif scheme == "shifted":
        x_span = float(np.max(x) - np.min(x))
        y_span = float(np.max(y) - np.min(y))
        x_use = x.copy()
        y_use = y.copy()
        x_cut = x_med + shift_frac * x_span
        y_cut = y_med + shift_frac * y_span

    elif scheme == "rotated":
        theta = math.radians(rotate_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x_centered = x - x_med
        y_centered = y - y_med

        x_use = x_centered * cos_t - y_centered * sin_t
        y_use = x_centered * sin_t + y_centered * cos_t

        x_cut = 0.0
        y_cut = 0.0

    else:
        raise ValueError(f"Unknown partition scheme: {scheme}")

    left = x_use <= x_cut
    upper = y_use <= y_cut

    codes = np.empty(x.shape[0], dtype=np.int8)
    codes[left & upper] = 0
    codes[(~left) & upper] = 1
    codes[left & (~upper)] = 2
    codes[(~left) & (~upper)] = 3

    info = {
        "scheme": scheme,
        "x_median": x_med,
        "y_median": y_med,
        "x_cut_used": float(x_cut),
        "y_cut_used": float(y_cut),
        "shift_frac": float(shift_frac),
        "rotate_deg": float(rotate_deg),
    }

    return codes, info, x_use, y_use


def codes_to_region_ids(codes: np.ndarray) -> np.ndarray:
    return np.asarray(REGION_ORDER, dtype=object)[codes]


def region_means_from_codes(
    X: np.ndarray,
    codes: np.ndarray,
    n_regions: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)

    counts = np.bincount(codes, minlength=n_regions).astype(float)
    means = np.full((n_regions, X.shape[1]), np.nan, dtype=float)

    for j in range(X.shape[1]):
        sums = np.bincount(codes, weights=X[:, j], minlength=n_regions).astype(float)
        valid = counts > 0
        means[valid, j] = sums[valid] / counts[valid]

    return means, counts.astype(int)


def safe_cv(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    if np.isclose(mu, 0.0):
        return np.nan
    return sd / abs(mu)


def mean_abs_pairwise_diff(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan

    diffs = []
    for i in range(vals.size):
        for j in range(i + 1, vals.size):
            diffs.append(abs(vals[i] - vals[j]))
    return float(np.mean(diffs))


def heterogeneity_from_region_means(
    dataset: str,
    sample_id: str,
    scheme: str,
    feature_cols: Sequence[str],
    region_means: np.ndarray,
    region_counts: np.ndarray,
    whole_means: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for j, feat in enumerate(feature_cols):
        vals = region_means[:, j]
        vals_valid = vals[np.isfinite(vals)]
        if vals_valid.size == 0:
            continue

        rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "scheme": scheme,
                "feature": feat,
                "n_regions": int(np.sum(np.isfinite(vals))),
                "region_range": float(np.nanmax(vals) - np.nanmin(vals)),
                "region_std": float(np.nanstd(vals, ddof=0)),
                "region_cv": float(safe_cv(vals)),
                "mean_abs_pairwise_diff": float(mean_abs_pairwise_diff(vals)),
                "mean_abs_deviation_from_global": float(
                    np.nanmean(np.abs(vals - whole_means[j]))
                ),
                "max_abs_deviation_from_global": float(
                    np.nanmax(np.abs(vals - whole_means[j]))
                ),
                "whole_mean": float(whole_means[j]),
                "min_region_mean": float(np.nanmin(vals)),
                "max_region_mean": float(np.nanmax(vals)),
                "min_region_count": int(np.min(region_counts)),
                "max_region_count": int(np.max(region_counts)),
            }
        )

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        return feature_df

    overall = {
        "dataset": dataset,
        "sample_id": sample_id,
        "scheme": scheme,
        "feature": "__overall__",
        "n_regions": int(feature_df["n_regions"].max()),
        "region_range": float(feature_df["region_range"].mean()),
        "region_std": float(feature_df["region_std"].mean()),
        "region_cv": float(feature_df["region_cv"].dropna().mean())
        if feature_df["region_cv"].notna().any()
        else np.nan,
        "mean_abs_pairwise_diff": float(feature_df["mean_abs_pairwise_diff"].mean()),
        "mean_abs_deviation_from_global": float(
            feature_df["mean_abs_deviation_from_global"].mean()
        ),
        "max_abs_deviation_from_global": float(
            feature_df["max_abs_deviation_from_global"].mean()
        ),
        "whole_mean": np.nan,
        "min_region_mean": np.nan,
        "max_region_mean": np.nan,
        "min_region_count": int(feature_df["min_region_count"].min()),
        "max_region_count": int(feature_df["max_region_count"].max()),
    }

    return pd.concat([feature_df, pd.DataFrame([overall])], ignore_index=True)


def region_means_long(
    dataset: str,
    sample_id: str,
    scheme: str,
    feature_cols: Sequence[str],
    region_means: np.ndarray,
    region_counts: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for ridx, region_id in enumerate(REGION_ORDER):
        for j, feat in enumerate(feature_cols):
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "scheme": scheme,
                    "region_id": region_id,
                    "region_label": REGION_LABELS[region_id],
                    "feature": feat,
                    "mean_value": float(region_means[ridx, j]),
                    "n_points_region": int(region_counts[ridx]),
                }
            )
    return pd.DataFrame(rows)


def whole_means_long(
    dataset: str,
    sample_id: str,
    feature_cols: Sequence[str],
    whole_means: np.ndarray,
    n_points: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "feature": feat,
                "whole_mean": float(whole_means[j]),
                "n_points_section": int(n_points),
            }
            for j, feat in enumerate(feature_cols)
        ]
    )


def metrics_to_long(
    wide_df: pd.DataFrame,
    source: str,
    perm_id: int | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in wide_df.iterrows():
        for metric in METRIC_COLS:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "scheme": row["scheme"],
                    "feature": row["feature"],
                    "source": source,
                    "perm_id": perm_id,
                    "metric_name": metric,
                    "value": row[metric],
                }
            )
    return pd.DataFrame(rows)


def run_shuffle_null(
    dataset: str,
    sample_id: str,
    X: np.ndarray,
    observed_codes: np.ndarray,
    feature_cols: Sequence[str],
    whole_means: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    obs_means, obs_counts = region_means_from_codes(X, observed_codes)
    obs_wide = heterogeneity_from_region_means(
        dataset=dataset,
        sample_id=sample_id,
        scheme="original",
        feature_cols=feature_cols,
        region_means=obs_means,
        region_counts=obs_counts,
        whole_means=whole_means,
    )
    obs_long = metrics_to_long(obs_wide, source="observed", perm_id=None)

    raw_parts = []
    for perm_id in range(n_perm):
        shuffled_codes = rng.permutation(observed_codes)
        perm_means, perm_counts = region_means_from_codes(X, shuffled_codes)

        perm_wide = heterogeneity_from_region_means(
            dataset=dataset,
            sample_id=sample_id,
            scheme="shuffle_null",
            feature_cols=feature_cols,
            region_means=perm_means,
            region_counts=perm_counts,
            whole_means=whole_means,
        )

        raw_parts.append(metrics_to_long(perm_wide, source="null", perm_id=perm_id))

    raw_long = pd.concat(raw_parts, ignore_index=True)

    summary_rows = []
    group_cols = ["dataset", "sample_id", "feature", "metric_name"]

    obs_lookup = obs_long[group_cols + ["value"]].rename(columns={"value": "observed_value"})

    for _, obs_row in obs_lookup.iterrows():
        mask = (
            (raw_long["dataset"] == obs_row["dataset"])
            & (raw_long["sample_id"] == obs_row["sample_id"])
            & (raw_long["feature"] == obs_row["feature"])
            & (raw_long["metric_name"] == obs_row["metric_name"])
        )

        null_vals = raw_long.loc[mask, "value"].dropna().to_numpy(dtype=float)
        observed_value = float(obs_row["observed_value"])

        if null_vals.size == 0:
            null_mean = np.nan
            null_sd = np.nan
            z_score = np.nan
            p_upper = np.nan
        else:
            null_mean = float(np.mean(null_vals))
            null_sd = float(np.std(null_vals, ddof=0))
            z_score = (
                float((observed_value - null_mean) / null_sd)
                if not np.isclose(null_sd, 0.0)
                else np.nan
            )
            p_upper = float((1 + np.sum(null_vals >= observed_value)) / (1 + null_vals.size))

        summary_rows.append(
            {
                "dataset": obs_row["dataset"],
                "sample_id": obs_row["sample_id"],
                "feature": obs_row["feature"],
                "metric_name": obs_row["metric_name"],
                "observed_value": observed_value,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z_score": z_score,
                "empirical_p_upper": p_upper,
                "n_perm": int(null_vals.size),
            }
        )

    return raw_long, pd.DataFrame(summary_rows)


def compute_partition_sensitivity(
    region_long_df: pd.DataFrame,
    heterogeneity_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    change_rows = []

    for (dataset, sample_id, feature), sub in heterogeneity_df.groupby(
        ["dataset", "sample_id", "feature"]
    ):
        orig = sub[sub["scheme"] == "original"]
        if orig.empty:
            continue

        orig = orig.iloc[0]

        for alt_scheme in ["shifted", "rotated"]:
            alt = sub[sub["scheme"] == alt_scheme]
            if alt.empty:
                continue

            alt = alt.iloc[0]

            for metric in METRIC_COLS:
                orig_val = orig[metric]
                alt_val = alt[metric]

                if pd.isna(orig_val) or pd.isna(alt_val):
                    delta = np.nan
                    pct_change = np.nan
                else:
                    delta = float(alt_val - orig_val)
                    pct_change = (
                        100.0 * delta / abs(float(orig_val))
                        if not np.isclose(float(orig_val), 0.0)
                        else np.nan
                    )

                change_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "feature": feature,
                        "scheme_ref": "original",
                        "scheme_alt": alt_scheme,
                        "metric_name": metric,
                        "original_value": float(orig_val) if pd.notna(orig_val) else np.nan,
                        "alt_value": float(alt_val) if pd.notna(alt_val) else np.nan,
                        "delta": delta,
                        "pct_change": pct_change,
                    }
                )

    corr_rows = []

    for (dataset, sample_id, feature), sub in region_long_df.groupby(
        ["dataset", "sample_id", "feature"]
    ):
        pivot = (
            sub.pivot_table(
                index="region_id",
                columns="scheme",
                values="mean_value",
                aggfunc="first",
            )
            .reindex(REGION_ORDER)
        )

        if "original" not in pivot.columns:
            continue

        orig_vec = pivot["original"].astype(float)

        for alt_scheme in ["shifted", "rotated"]:
            if alt_scheme not in pivot.columns:
                continue

            alt_vec = pivot[alt_scheme].astype(float)
            rho = orig_vec.corr(alt_vec, method="spearman")
            mad = float(np.nanmean(np.abs(orig_vec.to_numpy() - alt_vec.to_numpy())))

            corr_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "feature": feature,
                    "scheme_ref": "original",
                    "scheme_alt": alt_scheme,
                    "spearman_rho_regions": float(rho) if pd.notna(rho) else np.nan,
                    "mean_abs_region_shift": mad,
                }
            )

    return pd.DataFrame(change_rows), pd.DataFrame(corr_rows)


def sample_for_plot(df: pd.DataFrame, max_points: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    idx = rng.choice(np.arange(len(df)), size=max_points, replace=False)
    return df.iloc[idx].copy()


def plot_original_partitions(
    assignment_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outdir: Path,
) -> None:
    datasets = assignment_df["dataset"].unique().tolist()
    colors = {
        "Q1_UL": "#1f77b4",
        "Q2_UR": "#ff7f0e",
        "Q3_LL": "#2ca02c",
        "Q4_LR": "#d62728",
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)

    for ax, dataset in zip(axes.flatten(), datasets):
        sub = assignment_df[assignment_df["dataset"] == dataset]
        for region_id in REGION_ORDER:
            g = sub[sub["region_id"] == region_id]
            ax.scatter(g[x_col], g[y_col], s=2, alpha=0.45, color=colors[region_id], label=region_id)
        ax.set_title(f"{dataset}: median 2x2 partition")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(frameon=False, markerscale=4)

    fig.tight_layout()
    fig.savefig(outdir / "figure_original_median_partition_scatter.pdf")
    fig.savefig(outdir / "figure_original_median_partition_scatter.png", dpi=300)
    plt.close(fig)


def plot_region_heatmaps(region_long_df: pd.DataFrame, outdir: Path) -> None:
    sub = region_long_df[region_long_df["scheme"] == "original"].copy()
    datasets = sub["dataset"].unique().tolist()

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5), squeeze=False)

    for ax, dataset in zip(axes.flatten(), datasets):
        g = sub[sub["dataset"] == dataset]
        mat = (
            g.pivot_table(index="region_id", columns="feature", values="mean_value", aggfunc="first")
            .reindex(REGION_ORDER)
            .reindex(columns=FEATURE_COLS)
        )
        im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
        ax.set_title(f"{dataset}: regional bridge means")
        ax.set_xticks(np.arange(len(FEATURE_COLS)))
        ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(REGION_ORDER)))
        ax.set_yticklabels(REGION_ORDER)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(outdir / "figure_original_region_bridge_heatmaps.pdf")
    fig.savefig(outdir / "figure_original_region_bridge_heatmaps.png", dpi=300)
    plt.close(fig)


def plot_shuffle_null(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outdir: Path,
    metric_name: str = "mean_abs_pairwise_diff",
) -> None:
    raw = raw_df[
        (raw_df["feature"] == "__overall__")
        & (raw_df["metric_name"] == metric_name)
    ].copy()
    summary = summary_df[
        (summary_df["feature"] == "__overall__")
        & (summary_df["metric_name"] == metric_name)
    ].copy()

    datasets = raw["dataset"].unique().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)

    for ax, dataset in zip(axes.flatten(), datasets):
        vals = raw.loc[raw["dataset"] == dataset, "value"].dropna().to_numpy(dtype=float)
        ax.hist(vals, bins=30, alpha=0.75)

        s = summary[summary["dataset"] == dataset]
        if not s.empty:
            obs = float(s["observed_value"].iloc[0])
            pval = float(s["empirical_p_upper"].iloc[0])
            ax.axvline(obs, linestyle="--", linewidth=2)
            ax.set_title(f"{dataset}: {metric_name}\nobserved={obs:.4f}, p={pval:.4g}")
        else:
            ax.set_title(f"{dataset}: {metric_name}")

        ax.set_xlabel(metric_name)
        ax.set_ylabel("Permutation count")

    fig.tight_layout()
    fig.savefig(outdir / f"figure_shuffle_null_{metric_name}.pdf")
    fig.savefig(outdir / f"figure_shuffle_null_{metric_name}.png", dpi=300)
    plt.close(fig)


def plot_partition_sensitivity(heterogeneity_df: pd.DataFrame, outdir: Path) -> None:
    sub = heterogeneity_df[heterogeneity_df["feature"] == "__overall__"].copy()
    scheme_order = ["original", "shifted", "rotated"]
    metrics = ["region_range", "mean_abs_pairwise_diff"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)

    for ax, metric in zip(axes.flatten(), metrics):
        for dataset, g in sub.groupby("dataset"):
            g = g.set_index("scheme").reindex(scheme_order).reset_index()
            ax.plot(g["scheme"], g[metric], marker="o", label=dataset)

        ax.set_title(metric)
        ax.set_xlabel("Partition scheme")
        ax.set_ylabel("Metric value")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outdir / "figure_partition_sensitivity.pdf")
    fig.savefig(outdir / "figure_partition_sensitivity.png", dpi=300)
    plt.close(fig)


def write_summary(
    outpath: Path,
    dataset_infos: List[Dict[str, object]],
    heterogeneity_df: pd.DataFrame,
    shuffle_summary_df: pd.DataFrame,
    partition_change_df: pd.DataFrame,
    n_perm: int,
) -> None:
    lines = []

    lines.append("Within-section spatial validation summary")
    lines.append("=" * 70)
    lines.append(
        "Scope: primary CRC and breast analyses each use one tissue section. "
        "These results are within-section validation, not slide-level or patient-level benchmarking."
    )
    lines.append("")
    lines.append(f"Shuffle permutations: {n_perm}")
    lines.append("Primary partition: xy_median_quadrants")
    lines.append("")

    lines.append("Input sections")
    lines.append("-" * 70)
    for info in dataset_infos:
        lines.append(
            f"{info['dataset']}: sample_id={info['sample_id']}, "
            f"n_before={info['n_before']}, n_after={info['n_after']}, "
            f"removed_missing={info['n_removed_missing']}"
        )
    lines.append("")

    lines.append("Observed original-scheme overall heterogeneity")
    lines.append("-" * 70)
    obs = heterogeneity_df[
        (heterogeneity_df["scheme"] == "original")
        & (heterogeneity_df["feature"] == "__overall__")
    ]
    for _, row in obs.iterrows():
        lines.append(
            f"{row['dataset']}: region_range={row['region_range']:.6f}, "
            f"mean_abs_pairwise_diff={row['mean_abs_pairwise_diff']:.6f}, "
            f"mean_abs_deviation_from_global={row['mean_abs_deviation_from_global']:.6f}"
        )
    lines.append("")

    lines.append("Shuffle-null summary: overall mean_abs_pairwise_diff")
    lines.append("-" * 70)
    sub = shuffle_summary_df[
        (shuffle_summary_df["feature"] == "__overall__")
        & (shuffle_summary_df["metric_name"] == "mean_abs_pairwise_diff")
    ]
    for _, row in sub.iterrows():
        lines.append(
            f"{row['dataset']}: observed={row['observed_value']:.6f}, "
            f"null_mean={row['null_mean']:.6f}, z={row['z_score']:.6f}, "
            f"empirical_p_upper={row['empirical_p_upper']:.6g}"
        )
    lines.append("")

    lines.append("Partition sensitivity: overall mean_abs_pairwise_diff")
    lines.append("-" * 70)
    sub = partition_change_df[
        (partition_change_df["feature"] == "__overall__")
        & (partition_change_df["metric_name"] == "mean_abs_pairwise_diff")
    ]
    for _, row in sub.iterrows():
        lines.append(
            f"{row['dataset']} {row['scheme_alt']}: "
            f"original={row['original_value']:.6f}, alt={row['alt_value']:.6f}, "
            f"delta={row['delta']:.6f}, pct_change={row['pct_change']:.2f}%"
        )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    input_map = {
        "CRC": args.crc_csv,
        "Breast": args.breast_csv,
    }

    whole_parts = []
    region_parts = []
    heterogeneity_parts = []
    scheme_info_parts = []
    assignment_plot_parts = []
    shuffle_raw_parts = []
    shuffle_summary_parts = []
    dataset_infos = []

    cache = {}

    for dataset, path in input_map.items():
        df = load_core4_csv(path, dataset, args.x_col, args.y_col)
        sample_id = str(df["sample_id"].iloc[0]) if "sample_id" in df.columns else dataset

        dataset_infos.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "n_before": int(df.attrs["n_before"]),
                "n_after": int(df.attrs["n_after"]),
                "n_removed_missing": int(df.attrs["n_removed_missing"]),
            }
        )

        X = df[FEATURE_COLS].to_numpy(dtype=float)
        whole_means = X.mean(axis=0)

        whole_parts.append(
            whole_means_long(
                dataset=dataset,
                sample_id=sample_id,
                feature_cols=FEATURE_COLS,
                whole_means=whole_means,
                n_points=len(df),
            )
        )

        for scheme in ["original", "shifted", "rotated"]:
            codes, info, _, _ = assign_partition(
                x=df[args.x_col].to_numpy(dtype=float),
                y=df[args.y_col].to_numpy(dtype=float),
                scheme=scheme,
                shift_frac=args.shift_frac,
                rotate_deg=args.rotate_deg,
            )

            region_means, region_counts = region_means_from_codes(X, codes)

            region_parts.append(
                region_means_long(
                    dataset=dataset,
                    sample_id=sample_id,
                    scheme=scheme,
                    feature_cols=FEATURE_COLS,
                    region_means=region_means,
                    region_counts=region_counts,
                )
            )

            heterogeneity_parts.append(
                heterogeneity_from_region_means(
                    dataset=dataset,
                    sample_id=sample_id,
                    scheme=scheme,
                    feature_cols=FEATURE_COLS,
                    region_means=region_means,
                    region_counts=region_counts,
                    whole_means=whole_means,
                )
            )

            scheme_info = {
                "dataset": dataset,
                "sample_id": sample_id,
                **info,
                "count_Q1_UL": int(region_counts[0]),
                "count_Q2_UR": int(region_counts[1]),
                "count_Q3_LL": int(region_counts[2]),
                "count_Q4_LR": int(region_counts[3]),
            }
            scheme_info_parts.append(pd.DataFrame([scheme_info]))

            if scheme == "original":
                cache[dataset] = {
                    "sample_id": sample_id,
                    "X": X,
                    "codes": codes,
                    "whole_means": whole_means,
                }

                plot_df = df[[args.x_col, args.y_col]].copy()
                plot_df["dataset"] = dataset
                plot_df["sample_id"] = sample_id
                plot_df["region_id"] = codes_to_region_ids(codes)
                plot_df = sample_for_plot(plot_df, args.max_points_plot, rng)
                assignment_plot_parts.append(plot_df)

    whole_df = pd.concat(whole_parts, ignore_index=True)
    region_df = pd.concat(region_parts, ignore_index=True)
    heterogeneity_df = pd.concat(heterogeneity_parts, ignore_index=True)
    scheme_info_df = pd.concat(scheme_info_parts, ignore_index=True)
    assignment_plot_df = pd.concat(assignment_plot_parts, ignore_index=True)

    whole_df.to_csv(args.outdir / "within_section_whole_means.csv", index=False)
    region_df.to_csv(args.outdir / "within_section_region_means_by_scheme.csv", index=False)
    heterogeneity_df.to_csv(
        args.outdir / "within_section_heterogeneity_metrics_by_scheme.csv",
        index=False,
    )
    scheme_info_df.to_csv(args.outdir / "within_section_partition_scheme_info.csv", index=False)
    assignment_plot_df.to_csv(
        args.outdir / "within_section_original_partition_sampled_points.csv",
        index=False,
    )

    print("[OK] wrote whole, region, heterogeneity, scheme-info, and sampled partition outputs")

    for offset, dataset in enumerate(["CRC", "Breast"], start=1):
        c = cache[dataset]
        raw_long, summary_long = run_shuffle_null(
            dataset=dataset,
            sample_id=c["sample_id"],
            X=c["X"],
            observed_codes=c["codes"],
            feature_cols=FEATURE_COLS,
            whole_means=c["whole_means"],
            n_perm=args.n_perm,
            seed=args.seed + offset,
        )
        shuffle_raw_parts.append(raw_long)
        shuffle_summary_parts.append(summary_long)
        print(f"[OK] {dataset}: completed shuffle null with n_perm={args.n_perm}")

    shuffle_raw_df = pd.concat(shuffle_raw_parts, ignore_index=True)
    shuffle_summary_df = pd.concat(shuffle_summary_parts, ignore_index=True)

    shuffle_raw_df.to_csv(args.outdir / "within_section_shuffle_null_raw_long.csv", index=False)
    shuffle_summary_df.to_csv(args.outdir / "within_section_shuffle_null_summary.csv", index=False)

    partition_change_df, region_corr_df = compute_partition_sensitivity(region_df, heterogeneity_df)

    partition_change_df.to_csv(
        args.outdir / "within_section_partition_metric_changes.csv",
        index=False,
    )
    region_corr_df.to_csv(
        args.outdir / "within_section_partition_region_correlations.csv",
        index=False,
    )

    plot_original_partitions(assignment_plot_df, args.x_col, args.y_col, args.outdir)
    plot_region_heatmaps(region_df, args.outdir)
    plot_shuffle_null(shuffle_raw_df, shuffle_summary_df, args.outdir)
    plot_partition_sensitivity(heterogeneity_df, args.outdir)

    write_summary(
        outpath=args.outdir / "within_section_validation_summary.txt",
        dataset_infos=dataset_infos,
        heterogeneity_df=heterogeneity_df,
        shuffle_summary_df=shuffle_summary_df,
        partition_change_df=partition_change_df,
        n_perm=args.n_perm,
    )

    print("\nWithin-section spatial validation complete.")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
