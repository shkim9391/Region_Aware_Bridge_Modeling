from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


BRIDGE_OUTPUT_COLS = [
    "bridge_version",
    "bridge_model_name",
    "bridge_model_ckpt",
    "bridge_n_spots_used",
    "bridge_epi_like_mean",
    "bridge_epi_like_std",
    "bridge_epi_like_pos_frac",
    "bridge_fibroblast_mean",
    "bridge_fibroblast_std",
    "bridge_fibroblast_pos_frac",
    "bridge_smooth_myoepi_mean",
    "bridge_smooth_myoepi_std",
    "bridge_smooth_myoepi_pos_frac",
    "bridge_ECM_mean",
    "bridge_ECM_std",
]


def normalize_key_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index)

    s = df[col]

    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64").astype(str)

    return s.astype("string").str.strip()


def choose_join_keys(ou_df: pd.DataFrame, bridge_df: pd.DataFrame) -> List[str]:
    preferred = [
        ["dataset", "sample_id"],
        ["dataset", "patient_id", "timepoint_index", "sample_id"],
        ["sample_id"],
        ["slide_id"],
    ]
    for keys in preferred:
        if all(k in ou_df.columns for k in keys) and all(k in bridge_df.columns for k in keys):
            return keys
    raise ValueError(
        "Could not auto-detect join keys. "
        "Provide --join-keys explicitly, or make sure both files share one of: "
        "[dataset,sample_id], [dataset,patient_id,timepoint_index,sample_id], [sample_id], [slide_id]."
    )


def build_bridge_append_df(bridge_df: pd.DataFrame) -> pd.DataFrame:
    out = bridge_df.copy()

    rename_map = {
        "model_name": "bridge_model_name",
        "model_ckpt": "bridge_model_ckpt",
        "n_spots_used": "bridge_n_spots_used",
        "epi_like_prob_mean": "bridge_epi_like_mean",
        "epi_like_prob_std": "bridge_epi_like_std",
        "epi_like_pos_frac": "bridge_epi_like_pos_frac",
        "fibroblast_prob_mean": "bridge_fibroblast_mean",
        "fibroblast_prob_std": "bridge_fibroblast_std",
        "fibroblast_pos_frac": "bridge_fibroblast_pos_frac",
        "smooth_myoepi_prob_mean": "bridge_smooth_myoepi_mean",
        "smooth_myoepi_prob_std": "bridge_smooth_myoepi_std",
        "smooth_myoepi_pos_frac": "bridge_smooth_myoepi_pos_frac",
        "ECM_mean": "bridge_ECM_mean",
        "ECM_std": "bridge_ECM_std",
    }

    out = out.rename(columns=rename_map)

    for c in BRIDGE_OUTPUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out


def main():
    ap = argparse.ArgumentParser(description="Merge OU table with bridge_sample_state.csv")
    ap.add_argument("--ou", required=True, help="Existing OU input table CSV")
    ap.add_argument("--bridge", required=True, help="bridge_sample_state.csv")
    ap.add_argument("--out", required=True, help="Output ou_branching_bridge_input.csv")
    ap.add_argument(
        "--join-keys",
        nargs="+",
        default=None,
        help="Optional explicit join keys, e.g. --join-keys dataset sample_id",
    )
    ap.add_argument(
        "--allow-duplicate-bridge-keys",
        action="store_true",
        help="Allow duplicate rows in bridge file for the same join key (not recommended)",
    )
    args = ap.parse_args()

    ou_path = Path(args.ou)
    bridge_path = Path(args.bridge)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ou_df = pd.read_csv(ou_path)
    bridge_df = pd.read_csv(bridge_path)

    join_keys = args.join_keys if args.join_keys else choose_join_keys(ou_df, bridge_df)

    print("Join keys:", join_keys)

    # normalize join keys to reduce string/int mismatch issues
    for k in join_keys:
        ou_df[k] = normalize_key_col(ou_df, k)
        bridge_df[k] = normalize_key_col(bridge_df, k)

    bridge_df = build_bridge_append_df(bridge_df)

    keep_cols = join_keys + BRIDGE_OUTPUT_COLS
    bridge_small = bridge_df[keep_cols].copy()

    # check duplicate keys on bridge side
    dup_mask = bridge_small.duplicated(subset=join_keys, keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0 and not args.allow_duplicate_bridge_keys:
        dup_preview = bridge_small.loc[dup_mask, join_keys].head(10)
        raise ValueError(
            f"Bridge table has {n_dup} rows participating in duplicate join keys. "
            f"Resolve duplicates before merging, or use --allow-duplicate-bridge-keys.\n"
            f"Preview:\n{dup_preview.to_string(index=False)}"
        )

    merged = ou_df.merge(
        bridge_small,
        on=join_keys,
        how="left",
        validate="m:1" if not args.allow_duplicate_bridge_keys else "m:m",
    )

    # match diagnostics
    bridge_present = merged["bridge_version"].notna()
    n_matched = int(bridge_present.sum())
    n_total = int(len(merged))
    n_unmatched = n_total - n_matched

    print(f"OU rows: {n_total}")
    print(f"Matched bridge rows: {n_matched}")
    print(f"Unmatched OU rows: {n_unmatched}")

    # preserve OU columns order, append bridge block at end
    ou_cols = [c for c in ou_df.columns]
    final_cols = ou_cols + [c for c in BRIDGE_OUTPUT_COLS if c in merged.columns]
    merged = merged[final_cols].copy()

    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # print appended columns for sanity
    print("Appended bridge columns:")
    print(", ".join([c for c in BRIDGE_OUTPUT_COLS if c in merged.columns]))


if __name__ == "__main__":
    main()
