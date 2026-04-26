from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "04_region_summaries"
    / "combined_region_bridge_summary_crc_breast_core4_regions4.csv"
)

OUTDIR = PROJECT_ROOT / "outputs" / "06_design_matrices"
OUTDIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]

BRIDGE_MEAN_COLS = [
    "bridge_epi_like_mean",
    "bridge_fibroblast_mean",
    "bridge_smooth_myoepi_mean",
    "bridge_ECM_mean",
]

TARGET_SHORT_NAMES = {
    "bridge_epi_like_mean": "epi_like",
    "bridge_fibroblast_mean": "fibroblast",
    "bridge_smooth_myoepi_mean": "smooth_myoepi",
    "bridge_ECM_mean": "ECM",
}


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def add_design_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["dataset"] = out["dataset"].astype(str)
    out["region_id"] = out["region_id"].astype(str)

    out["observation_id"] = (
        out["dataset"].str.replace(" ", "_", regex=False)
        + "__"
        + out["region_id"]
    )

    out["dataset_indicator_breast"] = (out["dataset"] == "Breast").astype(int)

    # Region indicators with Q1_UL as reference.
    for region in REGION_ORDER[1:]:
        out[f"region_indicator_{region}"] = (out["region_id"] == region).astype(int)

    out["region_order_index"] = out["region_id"].map(
        {region: idx for idx, region in enumerate(REGION_ORDER)}
    )

    return out


def compute_standardization(df: pd.DataFrame, cols: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    stats_rows = []

    for col in cols:
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
                "z_feature": z_col,
                "mean": mu,
                "std_population_ddof0": sd,
                "n_fit": int(vals.notna().sum()),
                "constant_flag": bool(constant_flag),
            }
        )

    return out, pd.DataFrame(stats_rows)


def build_model_matrix(design_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build compact modeling table with standardized bridge features and
    dataset/region indicators.

    This is intentionally simple and transparent.
    """
    keep_cols = [
        "observation_id",
        "dataset",
        "region_id",
        "region_label",
        "region_scheme",
        "n_spots_used",
        "dataset_indicator_breast",
        "region_indicator_Q2_UR",
        "region_indicator_Q3_LL",
        "region_indicator_Q4_LR",
        "bridge_epi_like_mean_joint_z",
        "bridge_fibroblast_mean_joint_z",
        "bridge_smooth_myoepi_mean_joint_z",
        "bridge_ECM_mean_joint_z",
        "bridge_epi_like_mean",
        "bridge_fibroblast_mean",
        "bridge_smooth_myoepi_mean",
        "bridge_ECM_mean",
    ]

    keep_cols = [c for c in keep_cols if c in design_df.columns]
    return design_df[keep_cols].copy()


def write_readme(outpath: Path, design_df: pd.DataFrame) -> None:
    lines = []

    lines.append("Joint CRC--Breast region-aware bridge design matrix")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Purpose")
    lines.append("-" * 70)
    lines.append(
        "This folder contains the clean eight-row joint design matrix used for "
        "exploratory ridge and Bayesian modeling in the revised manuscript."
    )
    lines.append("")
    lines.append("Important interpretation")
    lines.append("-" * 70)
    lines.append(
        "The design matrix contains four median-quadrant regions from CRC and "
        "four median-quadrant regions from Breast. Because there are only eight "
        "region-level observations, downstream regression analyses should be "
        "interpreted as regularized exploratory summaries, not confirmatory "
        "population-level inference."
    )
    lines.append("")
    lines.append("Rows")
    lines.append("-" * 70)
    lines.append(f"Total observations: {len(design_df)}")
    for dataset, g in design_df.groupby("dataset", sort=False):
        lines.append(
            f"{dataset}: {len(g)} regions, total spots={int(g['n_spots_used'].sum())}"
        )
    lines.append("")
    lines.append("Primary bridge features")
    lines.append("-" * 70)
    for col in BRIDGE_MEAN_COLS:
        lines.append(f"- {col}")
    lines.append("")
    lines.append("Standardization")
    lines.append("-" * 70)
    lines.append(
        "Bridge features are standardized across the joint eight-row CRC--Breast "
        "design matrix using population standard deviation with ddof=0."
    )
    lines.append("")
    lines.append("Region encoding")
    lines.append("-" * 70)
    lines.append("Primary region scheme: xy_median_quadrants")
    lines.append("Region order: Q1_UL, Q2_UR, Q3_LL, Q4_LR")
    lines.append("Reference region for indicator encoding: Q1_UL")
    lines.append("Dataset indicator: Breast=1, CRC=0")
    lines.append("")

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = [
        "dataset",
        "sample_id",
        "region_id",
        "region_label",
        "region_scheme",
        "n_spots_used",
        *BRIDGE_MEAN_COLS,
    ]
    require_columns(df, required)

    # Keep the expected row order: CRC Q1-Q4, Breast Q1-Q4.
    df["region_id"] = pd.Categorical(df["region_id"], categories=REGION_ORDER, ordered=True)
    df = df.sort_values(["dataset", "region_id"]).reset_index(drop=True)

    # Ensure CRC first, Breast second.
    dataset_order = pd.CategoricalDtype(categories=["CRC", "Breast"], ordered=True)
    df["dataset"] = df["dataset"].astype(dataset_order)
    df = df.sort_values(["dataset", "region_id"]).reset_index(drop=True)
    df["dataset"] = df["dataset"].astype(str)

    design = add_design_labels(df)
    design, z_stats = compute_standardization(design, BRIDGE_MEAN_COLS)
    model_matrix = build_model_matrix(design)

    design_out = OUTDIR / "joint_region_bridge_design_matrix_crc_breast_core4.csv"
    model_out = OUTDIR / "joint_region_bridge_model_matrix_crc_breast_core4.csv"
    stats_out = OUTDIR / "joint_region_bridge_standardization_stats_crc_breast_core4.csv"
    readme_out = OUTDIR / "joint_region_bridge_design_matrix_readme.txt"

    design.to_csv(design_out, index=False)
    model_matrix.to_csv(model_out, index=False)
    z_stats.to_csv(stats_out, index=False)
    write_readme(readme_out, design)

    print(f"[OK] wrote {design_out}")
    print(f"[OK] wrote {model_out}")
    print(f"[OK] wrote {stats_out}")
    print(f"[OK] wrote {readme_out}")
    print("")
    print("Design matrix summary:")
    print(f"  rows: {len(design)}")
    for dataset, g in design.groupby("dataset", sort=False):
        print(f"  {dataset}: regions={len(g)}, spots={int(g['n_spots_used'].sum())}")

    print("")
    print("Bridge feature standardization:")
    print(z_stats.to_string(index=False))


if __name__ == "__main__":
    main()
