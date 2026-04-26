from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUTS = {
    "CRC": PROJECT_ROOT / "outputs" / "02_patch_indices" / "crc_patch_index_core4.csv",
    "Breast": PROJECT_ROOT / "outputs" / "02_patch_indices" / "breast_patch_index_core4.csv",
}

OUTDIR = PROJECT_ROOT / "outputs" / "04_region_summaries"
OUTDIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]

REGION_LABELS = {
    "Q1_UL": "upper_left",
    "Q2_UR": "upper_right",
    "Q3_LL": "lower_left",
    "Q4_LR": "lower_right",
}

CORE4_DIMS = [
    "epi_like",
    "fibroblast",
    "smooth_myoepi",
    "ECM",
]

BRIDGE_RENAME = {
    "epi_like": "bridge_epi_like",
    "fibroblast": "bridge_fibroblast",
    "smooth_myoepi": "bridge_smooth_myoepi",
    "ECM": "bridge_ECM",
}


def require_columns(df: pd.DataFrame, cols: List[str], dataset: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset} missing columns: {missing}")


def assign_median_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    x_cut = float(out["x0"].median())
    y_cut = float(out["y0"].median())

    left = out["x0"] <= x_cut
    upper = out["y0"] <= y_cut

    out["region_id"] = np.select(
        [
            left & upper,
            (~left) & upper,
            left & (~upper),
            (~left) & (~upper),
        ],
        REGION_ORDER,
        default="Q4_LR",
    )

    out["region_label"] = out["region_id"].map(REGION_LABELS)
    out["region_scheme"] = "xy_median_quadrants"
    out["region_x_cut"] = x_cut
    out["region_y_cut"] = y_cut

    return out


def summarize_numeric(x: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "q25": float(np.quantile(vals, 0.25)),
        "median": float(np.median(vals)),
        "q75": float(np.quantile(vals, 0.75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def positive_fraction(x: pd.Series, threshold: float = 0.5) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals >= threshold))


def aggregate_one(dataset: str, input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input for {dataset}: {input_csv}")

    usecols = [
        "dataset",
        "sample_id",
        "barcode",
        "x0",
        "y0",
        "x1",
        "y1",
        "image_path",
        *CORE4_DIMS,
    ]

    df = pd.read_csv(input_csv, usecols=usecols)
    require_columns(df, usecols, dataset)

    df["dataset"] = dataset

    for col in ["x0", "y0", "x1", "y1", *CORE4_DIMS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x0", "y0", *CORE4_DIMS]).copy()

    df = assign_median_quadrants(df)

    rows = []
    for region_id in REGION_ORDER:
        g = df.loc[df["region_id"] == region_id].copy()
        if g.empty:
            continue

        row = {
            "dataset": dataset,
            "sample_id": str(g["sample_id"].iloc[0]),
            "region_id": region_id,
            "region_label": REGION_LABELS[region_id],
            "region_scheme": "xy_median_quadrants",
            "region_x_cut": float(g["region_x_cut"].iloc[0]),
            "region_y_cut": float(g["region_y_cut"].iloc[0]),
            "n_spots_total": int(len(g)),
            "n_spots_used": int(len(g)),
            "n_unique_barcodes": int(g["barcode"].nunique()),
            "n_unique_image_paths": int(g["image_path"].nunique()),
            "x0_min": float(g["x0"].min()),
            "x0_max": float(g["x0"].max()),
            "y0_min": float(g["y0"].min()),
            "y0_max": float(g["y0"].max()),
        }

        for raw_col in CORE4_DIMS:
            bridge_col = BRIDGE_RENAME[raw_col]
            stats = summarize_numeric(g[raw_col])
            for stat_name, stat_val in stats.items():
                row[f"{bridge_col}_{stat_name}"] = stat_val

            if raw_col != "ECM":
                row[f"{bridge_col}_pos_frac_ge_0p5"] = positive_fraction(g[raw_col], threshold=0.5)

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(f"No region summaries created for {dataset}")

    out_csv = OUTDIR / f"{dataset.lower()}_region_bridge_summary_core4_regions4.csv"
    out.to_csv(out_csv, index=False)

    print(f"[OK] {dataset}: wrote {out_csv}")
    print(f"     regions={len(out)}, total spots used={int(out['n_spots_used'].sum())}")

    return out


def add_joint_standardization(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = master.copy()

    mean_cols = [
        "bridge_epi_like_mean",
        "bridge_fibroblast_mean",
        "bridge_smooth_myoepi_mean",
        "bridge_ECM_mean",
    ]

    stats_rows = []
    for col in mean_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        z_col = f"{col}_joint_z"

        constant_flag = (not np.isfinite(sd)) or np.isclose(sd, 0.0)
        if constant_flag:
            out[z_col] = 0.0
        else:
            out[z_col] = (vals - mu) / sd

        stats_rows.append(
            {
                "feature": col,
                "mean": mu,
                "std_population_ddof0": sd,
                "n_fit": int(vals.notna().sum()),
                "constant_flag": bool(constant_flag),
                "z_column": z_col,
            }
        )

    return out, pd.DataFrame(stats_rows)


def make_dataset_summary(master: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "bridge_epi_like_mean",
        "bridge_fibroblast_mean",
        "bridge_smooth_myoepi_mean",
        "bridge_ECM_mean",
    ]

    rows = []
    for dataset, g in master.groupby("dataset", sort=False):
        row = {
            "dataset": dataset,
            "n_regions": int(len(g)),
            "n_spots_used": int(g["n_spots_used"].sum()),
        }
        for col in mean_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    region_tables = []
    for dataset, input_csv in INPUTS.items():
        region_tables.append(aggregate_one(dataset, input_csv))

    master = pd.concat(region_tables, ignore_index=True, sort=False)

    master_z, z_stats = add_joint_standardization(master)
    dataset_summary = make_dataset_summary(master_z)

    combined_out = OUTDIR / "combined_region_bridge_summary_crc_breast_core4_regions4.csv"
    z_stats_out = OUTDIR / "region_bridge_standardization_stats_crc_breast_core4.csv"
    dataset_summary_out = OUTDIR / "dataset_region_bridge_summary_crc_breast_core4.csv"

    master_z.to_csv(combined_out, index=False)
    z_stats.to_csv(z_stats_out, index=False)
    dataset_summary.to_csv(dataset_summary_out, index=False)

    print(f"[OK] wrote {combined_out}")
    print(f"[OK] wrote {z_stats_out}")
    print(f"[OK] wrote {dataset_summary_out}")


if __name__ == "__main__":
    main()
