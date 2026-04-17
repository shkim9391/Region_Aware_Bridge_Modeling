from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]

TARGET_CANDIDATES: Dict[str, List[str]] = {
    "epi_like": [
        "epi_like_prob",
        "epi_like",
        "pred_epi_like",
        "yhat_epi_like",
    ],
    "fibroblast": [
        "fibroblast_prob",
        "fibroblast",
        "pred_fibroblast",
        "yhat_fibroblast",
    ],
    "smooth_myoepi": [
        "smooth_myoepi_prob",
        "smooth_myoepi",
        "pred_smooth_myoepi",
        "yhat_smooth_myoepi",
    ],
    "ECM": [
        "ECM",
        "ECM_pred",
        "pred_ECM",
        "yhat_ECM",
        "ECM_score",
        "ECM_prob",
    ],
}

X_CANDIDATES = ["x0", "x", "patch_x", "tile_x", "coord_x"]
Y_CANDIDATES = ["y0", "y", "patch_y", "tile_y", "coord_y"]

SAMPLE_ID_CANDIDATES = ["sample_id", "slide_id", "image_id"]
PATIENT_ID_CANDIDATES = ["patient_id", "case_id"]
TIMEPOINT_INDEX_CANDIDATES = ["timepoint_index"]
TIMEPOINT_LABEL_CANDIDATES = ["timepoint_label"]
TIME_FROM_BASELINE_CANDIDATES = ["time_from_baseline_days"]
SLIDE_ID_CANDIDATES = ["slide_id", "image_path"]


def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def require_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    col = pick_first_present(df, candidates)
    if col is None:
        raise ValueError(
            f"Could not find a column for {label}. "
            f"Tried: {', '.join(candidates)}. "
            f"Available columns: {', '.join(df.columns)}"
        )
    return col


def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
    }


def positive_fraction(series: pd.Series, threshold: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return np.nan
    return float(np.mean(x >= threshold))

def maybe_sigmoid(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    finite = x[np.isfinite(x)]
    if finite.empty:
        return x

    # If values already look like probabilities, leave them alone.
    if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return x

    # Otherwise treat as logits and convert to probabilities.
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))

def coerce_scalar(value, default):
    if pd.isna(value):
        return default
    return value


def assign_quadrants(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    out = df.copy()

    x_mid = (float(out[xcol].min()) + float(out[xcol].max())) / 2.0
    y_mid = (float(out[ycol].min()) + float(out[ycol].max())) / 2.0

    left = out[xcol] <= x_mid
    upper = out[ycol] <= y_mid

    out["region_id"] = np.select(
        [
            left & upper,
            (~left) & upper,
            left & (~upper),
            (~left) & (~upper),
        ],
        [
            "Q1_UL",
            "Q2_UR",
            "Q3_LL",
            "Q4_LR",
        ],
        default="Q4_LR",
    )

    out["region_label"] = out["region_id"].map(
        {
            "Q1_UL": "upper_left",
            "Q2_UR": "upper_right",
            "Q3_LL": "lower_left",
            "Q4_LR": "lower_right",
        }
    )
    out["region_scheme"] = "xy_midpoint_quadrants"
    out["region_x_mid"] = x_mid
    out["region_y_mid"] = y_mid
    return out


def infer_sample_id(df: pd.DataFrame, sample_id_default: str) -> str:
    sample_col = pick_first_present(df, SAMPLE_ID_CANDIDATES)
    if sample_col is not None:
        vals = df[sample_col].dropna().astype(str).unique().tolist()
        if len(vals) >= 1:
            return vals[0]

    if "image_path" in df.columns:
        vals = df["image_path"].dropna().astype(str).unique().tolist()
        if len(vals) >= 1:
            return Path(vals[0]).stem

    return sample_id_default


def infer_patient_id(df: pd.DataFrame) -> Optional[str]:
    patient_col = pick_first_present(df, PATIENT_ID_CANDIDATES)
    if patient_col is None:
        return None
    vals = df[patient_col].dropna().astype(str).unique().tolist()
    if len(vals) == 0:
        return None
    return vals[0]


def infer_timepoint_index(df: pd.DataFrame) -> int:
    col = pick_first_present(df, TIMEPOINT_INDEX_CANDIDATES)
    if col is None:
        return 0
    vals = df[col].dropna().tolist()
    if len(vals) == 0:
        return 0
    return int(vals[0])


def infer_timepoint_label(df: pd.DataFrame) -> str:
    col = pick_first_present(df, TIMEPOINT_LABEL_CANDIDATES)
    if col is None:
        return "Baseline"
    vals = df[col].dropna().astype(str).tolist()
    if len(vals) == 0:
        return "Baseline"
    return vals[0]


def infer_time_from_baseline_days(df: pd.DataFrame):
    col = pick_first_present(df, TIME_FROM_BASELINE_CANDIDATES)
    if col is None:
        return np.nan
    vals = df[col].dropna().tolist()
    if len(vals) == 0:
        return np.nan
    return vals[0]


def infer_slide_id_list(df: pd.DataFrame) -> str:
    col = pick_first_present(df, SLIDE_ID_CANDIDATES)
    if col is None:
        return ""
    vals = df[col].dropna().astype(str).unique().tolist()
    return "|".join(vals)


def build_region_bridge_rows(
    pred_csv: Path,
    dataset_default: str,
    disease_default: str,
    bridge_version: str,
    model_name: str,
    model_ckpt: str,
    bce_threshold: float,
    sample_id_default: str,
) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)

    xcol = require_column(df, X_CANDIDATES, "x coordinate")
    ycol = require_column(df, Y_CANDIDATES, "y coordinate")

    detected = {
        target: require_column(df, candidates, target)
        for target, candidates in TARGET_CANDIDATES.items()
    }

    dataset = dataset_default
    if "dataset" in df.columns:
        vals = df["dataset"].dropna().astype(str).unique().tolist()
        if len(vals) >= 1:
            dataset = vals[0]

    disease = disease_default
    if "disease" in df.columns:
        vals = df["disease"].dropna().astype(str).unique().tolist()
        if len(vals) >= 1:
            disease = vals[0]

    sample_id = infer_sample_id(df, sample_id_default=sample_id_default)
    patient_id = infer_patient_id(df)
    timepoint_index = infer_timepoint_index(df)
    timepoint_label = infer_timepoint_label(df)
    time_from_baseline_days = infer_time_from_baseline_days(df)
    slide_id_list = infer_slide_id_list(df)

    df = assign_quadrants(df, xcol=xcol, ycol=ycol)

    rows = []
    for region_id in REGION_ORDER:
        g = df.loc[df["region_id"] == region_id].copy()
        if g.empty:
            continue

        epi_vals = maybe_sigmoid(g[detected["epi_like"]])
        fib_vals = maybe_sigmoid(g[detected["fibroblast"]])
        smy_vals = maybe_sigmoid(g[detected["smooth_myoepi"]])
        ecm_vals = pd.to_numeric(g[detected["ECM"]], errors="coerce")

        epi_stats = summarize_numeric(epi_vals)
        fib_stats = summarize_numeric(fib_vals)
        smy_stats = summarize_numeric(smy_vals)
        ecm_stats = summarize_numeric(ecm_vals)

        row = {
            "bridge_version": bridge_version,
            "model_name": model_name,
            "model_ckpt": model_ckpt,
            "dataset": dataset,
            "disease": disease,
            "patient_id": patient_id,
            "sample_id": sample_id,
            "timepoint_index": timepoint_index,
            "timepoint_label": timepoint_label,
            "time_from_baseline_days": time_from_baseline_days,
            "region_id": region_id,
            "region_label": g["region_label"].iloc[0],
            "region_scheme": g["region_scheme"].iloc[0],
            "slide_id_list": slide_id_list,
            "n_slides": len(slide_id_list.split("|")) if slide_id_list else 1,
            "aggregation_level": "quadrant",
            "bce_positive_threshold": bce_threshold,
            "n_spots_total": int(len(g)),
            "n_spots_used": int(len(g)),
            "x0_min": float(g[xcol].min()),
            "x0_max": float(g[xcol].max()),
            "y0_min": float(g[ycol].min()),
            "y0_max": float(g[ycol].max()),
            "epi_like_prob_mean": epi_stats["mean"],
            "epi_like_prob_std": epi_stats["std"],
            "epi_like_prob_q25": epi_stats["q25"],
            "epi_like_prob_median": epi_stats["median"],
            "epi_like_prob_q75": epi_stats["q75"],
            "epi_like_pos_frac": positive_fraction(epi_vals, bce_threshold),
            "fibroblast_prob_mean": fib_stats["mean"],
            "fibroblast_prob_std": fib_stats["std"],
            "fibroblast_prob_q25": fib_stats["q25"],
            "fibroblast_prob_median": fib_stats["median"],
            "fibroblast_prob_q75": fib_stats["q75"],
            "fibroblast_pos_frac": positive_fraction(fib_vals, bce_threshold),
            "smooth_myoepi_prob_mean": smy_stats["mean"],
            "smooth_myoepi_prob_std": smy_stats["std"],
            "smooth_myoepi_prob_q25": smy_stats["q25"],
            "smooth_myoepi_prob_median": smy_stats["median"],
            "smooth_myoepi_prob_q75": smy_stats["q75"],
            "smooth_myoepi_pos_frac": positive_fraction(smy_vals, bce_threshold),
            "ECM_mean": ecm_stats["mean"],
            "ECM_std": ecm_stats["std"],
            "ECM_q25": ecm_stats["q25"],
            "ECM_median": ecm_stats["median"],
            "ECM_q75": ecm_stats["q75"],
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No quadrant rows were created from {pred_csv}")
    return out


def add_joint_z(master: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = master.copy()
    for c in cols:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        zcol = f"{c}_joint_z"
        if (not np.isfinite(sd)) or sd == 0.0:
            out[zcol] = 0.0
        else:
            out[zcol] = (out[c] - mu) / sd
    return out


def make_combined_tables(
    crc_regions: pd.DataFrame,
    breast_regions: pd.DataFrame,
    combined_outdir: Path,
) -> None:
    combined_outdir.mkdir(parents=True, exist_ok=True)

    def to_bridge_table(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bridge_dataset"] = out["dataset"]
        out = out.rename(
            columns={
                "epi_like_prob_mean": "bridge_epi_like_mean",
                "fibroblast_prob_mean": "bridge_fibroblast_mean",
                "smooth_myoepi_prob_mean": "bridge_smooth_myoepi_mean",
                "ECM_mean": "bridge_ECM_mean",
            }
        )
        return out

    master = pd.concat(
        [to_bridge_table(crc_regions), to_bridge_table(breast_regions)],
        ignore_index=True,
        sort=False,
    )

    bridge_cols = [
        "bridge_epi_like_mean",
        "bridge_fibroblast_mean",
        "bridge_smooth_myoepi_mean",
        "bridge_ECM_mean",
    ]

    master = add_joint_z(master, bridge_cols)

    stats_rows = []
    for c in bridge_cols:
        stats_rows.append(
            {
                "feature": c,
                "mean": float(master[c].mean()),
                "std": float(master[c].std(ddof=0)),
                "n_fit": int(master[c].notna().sum()),
                "constant_flag": bool(
                    (not np.isfinite(master[c].std(ddof=0)))
                    or (master[c].std(ddof=0) == 0.0)
                ),
            }
        )
    joint_stats = pd.DataFrame(stats_rows)

    summary_rows = []
    for cohort, g in master.groupby("bridge_dataset", sort=False):
        row = {"bridge_dataset": cohort, "n_rows": int(len(g))}
        for c in bridge_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=0))
            row[f"{c}_median"] = float(g[c].median())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    delta_rows = []
    idx = summary.set_index("bridge_dataset")
    if {"CRC", "Breast"}.issubset(idx.index):
        for c in bridge_cols:
            delta_rows.append(
                {
                    "feature": c,
                    "crc_mean": float(idx.loc["CRC", f"{c}_mean"]),
                    "breast_mean": float(idx.loc["Breast", f"{c}_mean"]),
                    "crc_minus_breast": float(
                        idx.loc["CRC", f"{c}_mean"] - idx.loc["Breast", f"{c}_mean"]
                    ),
                }
            )
    delta = pd.DataFrame(delta_rows)

    keep_cols = [
        "bridge_dataset",
        "dataset",
        "sample_id",
        "timepoint_index",
        "timepoint_label",
        "region_id",
        "region_label",
        "n_spots_used",
        "bridge_epi_like_mean",
        "bridge_fibroblast_mean",
        "bridge_smooth_myoepi_mean",
        "bridge_ECM_mean",
        "bridge_epi_like_mean_joint_z",
        "bridge_fibroblast_mean_joint_z",
        "bridge_smooth_myoepi_mean_joint_z",
        "bridge_ECM_mean_joint_z",
    ]
    keep_cols = [c for c in keep_cols if c in master.columns]
    multirow = master[keep_cols].copy()

    master.to_csv(
        combined_outdir / "bridge_master_allcols_crc_breast_core4_regions4.csv",
        index=False,
    )
    multirow.to_csv(
        combined_outdir / "bridge_multirow_table_crc_breast_core4_regions4.csv",
        index=False,
    )
    joint_stats.to_csv(
        combined_outdir / "bridge_joint_standardization_stats_crc_breast_core4_regions4.csv",
        index=False,
    )
    summary.to_csv(
        combined_outdir / "bridge_dataset_summary_crc_vs_breast_core4_regions4.csv",
        index=False,
    )
    delta.to_csv(
        combined_outdir / "bridge_feature_delta_crc_minus_breast_core4_regions4.csv",
        index=False,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Split CRC and Breast spot-level prediction CSVs into 4 quadrants each and aggregate to 8 total bridge rows."
    )

    ap.add_argument(
        "--crc-pred",
        default="output/output_crc_fixed_dataset_core4_from_easy_v1/val_predictions_crc_fixed_dataset_core4_from_easy_v1_best.csv",
    )
    ap.add_argument(
        "--breast-pred",
        default="output/output_breast_fixed_dataset_core4_from_easy_v1/val_predictions_breast_fixed_dataset_core4_from_easy_v1_best.csv",
    )

    ap.add_argument(
        "--crc-out",
        default="output/output_crc_fixed_dataset_core4_from_easy_v1/bridge_sample_state_crc_core4_regions4.csv",
    )
    ap.add_argument(
        "--breast-out",
        default="output/output_breast_fixed_dataset_core4_from_easy_v1/bridge_sample_state_breast_core4_regions4.csv",
    )
    ap.add_argument(
        "--combined-outdir",
        default="output/output_crc_breast_bridge_compare_core4_regions4",
    )

    ap.add_argument("--bridge-version", default="core4_from_easy_v1_v1_regions4")
    ap.add_argument("--model-name", default="resnet18_core4_from_easy_v1")
    ap.add_argument(
        "--crc-model-ckpt",
        default="output/output_crc_fixed_dataset_core4_from_easy_v1/wsi_to_z_best.pt",
    )
    ap.add_argument(
        "--breast-model-ckpt",
        default="output/output_crc_fixed_dataset_core4_from_easy_v1/wsi_to_z_best.pt",
    )
    ap.add_argument("--bce-threshold", type=float, default=0.5)

    ap.add_argument("--crc-sample-id-default", default="Datasets_Visium_DH_FF_tissue_hires_image")
    ap.add_argument("--breast-sample-id-default", default="Datasets_Visium_DH_FF_tissue_hires_image")

    args = ap.parse_args()

    crc_pred = Path(args.crc_pred)
    breast_pred = Path(args.breast_pred)
    crc_out = Path(args.crc_out)
    breast_out = Path(args.breast_out)
    combined_outdir = Path(args.combined_outdir)

    crc_out.parent.mkdir(parents=True, exist_ok=True)
    breast_out.parent.mkdir(parents=True, exist_ok=True)
    combined_outdir.mkdir(parents=True, exist_ok=True)

    crc_regions = build_region_bridge_rows(
        pred_csv=crc_pred,
        dataset_default="CRC",
        disease_default="CRC",
        bridge_version=args.bridge_version,
        model_name=args.model_name,
        model_ckpt=args.crc_model_ckpt,
        bce_threshold=args.bce_threshold,
        sample_id_default=args.crc_sample_id_default,
    )

    breast_regions = build_region_bridge_rows(
        pred_csv=breast_pred,
        dataset_default="Breast",
        disease_default="Breast",
        bridge_version=args.bridge_version,
        model_name=args.model_name,
        model_ckpt=args.breast_model_ckpt,
        bce_threshold=args.bce_threshold,
        sample_id_default=args.breast_sample_id_default,
    )

    crc_regions.to_csv(crc_out, index=False)
    breast_regions.to_csv(breast_out, index=False)

    make_combined_tables(
        crc_regions=crc_regions,
        breast_regions=breast_regions,
        combined_outdir=combined_outdir,
    )

    print("Saved:")
    print(" -", crc_out)
    print(" -", breast_out)
    print(" -", combined_outdir / "bridge_master_allcols_crc_breast_core4_regions4.csv")
    print(" -", combined_outdir / "bridge_multirow_table_crc_breast_core4_regions4.csv")
    print(" -", combined_outdir / "bridge_joint_standardization_stats_crc_breast_core4_regions4.csv")
    print(" -", combined_outdir / "bridge_dataset_summary_crc_vs_breast_core4_regions4.csv")
    print(" -", combined_outdir / "bridge_feature_delta_crc_minus_breast_core4_regions4.csv")


if __name__ == "__main__":
    main()
