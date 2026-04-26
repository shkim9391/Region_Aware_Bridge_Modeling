from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEGACY_OUT = Path(
    "/Datasets_Visium_DH_FF/output"
)

INPUTS = {
    "CRC": LEGACY_OUT / "crc_patch_index.csv",
    "Breast": LEGACY_OUT / "breast_patch_index.csv",
}

OUTDIR = PROJECT_ROOT / "outputs" / "02_patch_indices"
OUTDIR.mkdir(parents=True, exist_ok=True)

COMPOSITION_DIMS = [
    "epi_like",
    "fibroblast",
    "endothelial",
    "myeloid",
    "lymphoid_plasma",
    "smooth_myoepi",
    "stressed",
]

PROGRAM_DIMS = [
    "IFNG",
    "IFN1",
    "CYTOTOX",
    "AP_MHC",
    "NFKB",
    "PROLIF",
    "HYPOXIA",
    "EMT",
    "OXPHOS",
    "UPR",
    "ECM",
    "ANGIO",
]

QC_DIMS = [
    "log_total_counts",
    "log_n_genes",
]

FULL21_DIMS = COMPOSITION_DIMS + PROGRAM_DIMS + QC_DIMS

CORE4_DIMS = [
    "epi_like",
    "fibroblast",
    "smooth_myoepi",
    "ECM",
]

REQUIRED_METADATA = [
    "barcode",
    "x_px",
    "y_px",
    "x0",
    "y0",
    "x1",
    "y1",
    "dataset",
    "image_path",
]


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def derive_sample_id(image_path_series: pd.Series, fallback: str) -> str:
    paths = image_path_series.dropna().astype(str).unique().tolist()
    if len(paths) == 1:
        return Path(paths[0]).stem
    if len(paths) > 1:
        return f"{fallback}_multiimage"
    return fallback


def clean_one(dataset_label: str, input_csv: Path) -> Dict[str, object]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV for {dataset_label}: {input_csv}")

    df = pd.read_csv(input_csv)

    require_columns(df, REQUIRED_METADATA + FULL21_DIMS, dataset_label)

    n_before = len(df)

    # Standardize dataset label.
    df["dataset"] = dataset_label

    # Add sample_id from image path.
    df["sample_id"] = derive_sample_id(df["image_path"], fallback=dataset_label)

    # Coerce numeric columns.
    numeric_cols = ["x_px", "y_px", "x0", "y0", "x1", "y1"] + FULL21_DIMS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing coordinates or core bridge features.
    required_nonmissing = ["barcode", "x0", "y0"] + CORE4_DIMS
    n_missing_before = int(df[required_nonmissing].isna().any(axis=1).sum())
    df = df.dropna(subset=required_nonmissing).copy()
    n_after = len(df)

    # Reorder columns.
    metadata_cols = [
        "dataset",
        "sample_id",
        "barcode",
        "x_px",
        "y_px",
        "x0",
        "y0",
        "x1",
        "y1",
        "image_path",
    ]

    full_cols = metadata_cols + FULL21_DIMS
    core_cols = metadata_cols + CORE4_DIMS

    full_df = df[full_cols].copy()
    core_df = df[core_cols].copy()

    prefix = dataset_label.lower()

    full_out = OUTDIR / f"{prefix}_patch_index_full21.csv"
    core_out = OUTDIR / f"{prefix}_patch_index_core4.csv"

    full_df.to_csv(full_out, index=False)
    core_df.to_csv(core_out, index=False)

    # Summaries.
    summary: Dict[str, object] = {
        "dataset": dataset_label,
        "input_csv": str(input_csv),
        "sample_id": str(df["sample_id"].iloc[0]) if len(df) else "",
        "n_rows_before_filter": int(n_before),
        "n_rows_after_filter": int(n_after),
        "n_rows_removed_missing_required": int(n_before - n_after),
        "n_rows_with_missing_required_before_filter": n_missing_before,
        "n_unique_barcodes": int(df["barcode"].nunique()),
        "n_unique_image_paths": int(df["image_path"].nunique()),
        "x0_min": float(df["x0"].min()),
        "x0_max": float(df["x0"].max()),
        "y0_min": float(df["y0"].min()),
        "y0_max": float(df["y0"].max()),
        "full21_output_csv": str(full_out),
        "core4_output_csv": str(core_out),
    }

    for dim in CORE4_DIMS:
        vals = pd.to_numeric(df[dim], errors="coerce")
        summary[f"{dim}_mean"] = float(vals.mean())
        summary[f"{dim}_std"] = float(vals.std(ddof=0))
        summary[f"{dim}_min"] = float(vals.min())
        summary[f"{dim}_max"] = float(vals.max())

    print(f"[OK] {dataset_label}: wrote {full_out}")
    print(f"[OK] {dataset_label}: wrote {core_out}")
    print(f"     rows before={n_before}, after={n_after}, removed={n_before - n_after}")

    return summary


def write_column_inventory() -> None:
    rows = []
    for group_name, cols in [
        ("metadata", REQUIRED_METADATA),
        ("composition", COMPOSITION_DIMS),
        ("program", PROGRAM_DIMS),
        ("qc", QC_DIMS),
        ("core4", CORE4_DIMS),
    ]:
        for col in cols:
            rows.append({"column_group": group_name, "column": col})

    out_csv = OUTDIR / "patch_index_columns.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


def main() -> None:
    write_column_inventory()

    summaries = []
    for dataset_label, input_csv in INPUTS.items():
        summaries.append(clean_one(dataset_label, input_csv))

    summary_out = OUTDIR / "patch_index_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_out, index=False)
    print(f"[OK] wrote {summary_out}")


if __name__ == "__main__":
    main()
