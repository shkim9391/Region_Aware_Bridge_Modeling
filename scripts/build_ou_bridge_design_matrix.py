from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BRIDGE_MEAN_COLS = [
    "bridge_epi_like_mean",
    "bridge_fibroblast_mean",
    "bridge_smooth_myoepi_mean",
    "bridge_ECM_mean",
]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fit_standardization_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = safe_numeric(df[col]) if col in df.columns else pd.Series(dtype=float)
        s = s.dropna()

        n_fit = int(len(s))
        if n_fit == 0:
            mean = np.nan
            std = np.nan
            constant_flag = True
        else:
            mean = float(s.mean())
            std = float(s.std(ddof=0))
            constant_flag = (not np.isfinite(std)) or std == 0.0

        rows.append(
            {
                "feature": col,
                "mean": mean,
                "std": std,
                "n_fit": n_fit,
                "constant_flag": bool(constant_flag),
            }
        )
    return pd.DataFrame(rows)


def apply_standardization(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    for _, row in stats_df.iterrows():
        col = str(row["feature"])
        zcol = f"{col}_z"

        if col not in out.columns:
            out[zcol] = np.nan
            continue

        s = safe_numeric(out[col])
        mean = row["mean"]
        std = row["std"]
        constant_flag = bool(row["constant_flag"])

        if not np.isfinite(mean) or constant_flag or not np.isfinite(std) or std == 0:
            # For constant / unavailable reference stats, set finite rows to 0.0
            z = pd.Series(np.nan, index=out.index, dtype=float)
            mask = s.notna()
            z.loc[mask] = 0.0
            out[zcol] = z
        else:
            out[zcol] = (s - mean) / std

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build OU-ready design matrix by adding standardized bridge _z covariates."
    )
    ap.add_argument(
        "--in",
        dest="input_csv",
        required=True,
        help="Input ou_branching_bridge_input.csv",
    )
    ap.add_argument(
        "--out",
        dest="output_csv",
        required=True,
        help="Output design matrix CSV with *_z columns appended",
    )
    ap.add_argument(
        "--stats-out",
        dest="stats_out",
        required=True,
        help="Output CSV containing normalization stats",
    )
    ap.add_argument(
        "--fit-on",
        dest="fit_on_csv",
        default=None,
        help=(
            "Optional CSV used to fit standardization stats. "
            "If omitted, stats are fit on the input CSV itself."
        ),
    )
    ap.add_argument(
        "--require-cols",
        action="store_true",
        help="Raise an error if any required bridge columns are missing.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_out)
    fit_on_path = Path(args.fit_on_csv) if args.fit_on_csv else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    in_df = pd.read_csv(input_path)

    fit_df = pd.read_csv(fit_on_path) if fit_on_path else in_df.copy()

    missing = [c for c in BRIDGE_MEAN_COLS if c not in fit_df.columns or c not in in_df.columns]
    if missing and args.require_cols:
        raise ValueError(f"Missing required bridge columns: {missing}")

    stats_df = fit_standardization_stats(fit_df, BRIDGE_MEAN_COLS)
    out_df = apply_standardization(in_df, stats_df)

    # Keep original columns first, append _z columns at the end
    z_cols = [f"{c}_z" for c in BRIDGE_MEAN_COLS]
    existing_z = [c for c in z_cols if c in out_df.columns]
    ordered_cols = [c for c in in_df.columns] + [c for c in existing_z if c not in in_df.columns]
    out_df = out_df[ordered_cols].copy()

    out_df.to_csv(output_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved design matrix: {output_path}")
    print(f"Saved standardization stats: {stats_path}")
    print("\nStandardized columns:")
    for c in existing_z:
        print(" -", c)

    print("\nStats preview:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
