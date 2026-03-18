from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


def pick_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_dataset_col(df: pd.DataFrame) -> str:
    col = pick_first_present(df, ["bridge_dataset", "dataset"])
    if col is None:
        raise ValueError(
            "Could not find a cohort column. Tried: bridge_dataset, dataset"
        )
    return col


def infer_region_col(df: pd.DataFrame) -> str:
    col = pick_first_present(df, ["region_id", "region_label"])
    if col is None:
        raise ValueError(
            "Could not find a region column. Tried: region_id, region_label"
        )
    return col


def infer_bridge_z_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("bridge_") and c.endswith("_z")]
    if not cols:
        raise ValueError(
            "Could not find any standardized bridge columns ending in '_z'."
        )
    return cols


def build_feature_matrix(
    df: pd.DataFrame,
    y_col: str,
    dataset_col: str,
    region_col: str,
    x_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if y_col not in df.columns:
        raise ValueError(f"Target column not found: {y_col}")

    if x_cols is None or len(x_cols) == 0:
        z_cols = infer_bridge_z_cols(df)
        x_cols = [c for c in z_cols if c != y_col]

    missing = [c for c in x_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {missing}")

    X_num = df[x_cols].apply(pd.to_numeric, errors="coerce")
    X_cat = pd.get_dummies(
        df[[dataset_col, region_col]].astype(str),
        drop_first=True,
        dtype=float,
    )

    X = pd.concat([X_num, X_cat], axis=1)
    y = pd.to_numeric(df[y_col], errors="coerce")

    keep_mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)

    return X, y, x_cols


def run_loocv(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.full(shape=len(y), fill_value=np.nan, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)
        preds[test_idx[0]] = float(model.predict(X_test)[0])

    return preds


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.isclose(np.std(y_true, ddof=0), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit a simple region-aware ridge model on a joint OU bridge design matrix."
    )
    ap.add_argument(
        "--in",
        dest="in_csv",
        default="output/output_crc_breast_bridge_compare_core4_regions4/ou_branching_bridge_design_matrix_crc_breast_core4_regions4.csv",
        help="Input joint design matrix CSV",
    )
    ap.add_argument(
        "--y",
        default="bridge_ECM_mean_z",
        help="Target column to model",
    )
    ap.add_argument(
        "--x-cols",
        nargs="*",
        default=None,
        help="Optional explicit predictor columns; default = all bridge *_z columns except y",
    )
    ap.add_argument(
        "--dataset-col",
        default=None,
        help="Optional cohort column; default = infer bridge_dataset or dataset",
    )
    ap.add_argument(
        "--region-col",
        default=None,
        help="Optional region column; default = infer region_id or region_label",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Ridge penalty strength",
    )
    ap.add_argument(
        "--outdir",
        default="output/output_crc_breast_bridge_compare_core4_regions4/region_model_ecm",
        help="Output directory",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    dataset_col = args.dataset_col or infer_dataset_col(df)
    region_col = args.region_col or infer_region_col(df)

    X, y, used_x_cols = build_feature_matrix(
        df=df,
        y_col=args.y,
        dataset_col=dataset_col,
        region_col=region_col,
        x_cols=args.x_cols,
    )

    model = Ridge(alpha=args.alpha, fit_intercept=True)
    model.fit(X, y)
    pred_full = model.predict(X)
    pred_loocv = run_loocv(X, y, alpha=args.alpha)

    # Row IDs for readable output
    id_cols = [
        c for c in ["bridge_dataset", "dataset", "sample_id", "region_id", "region_label"]
        if c in df.columns
    ]
    ids = df.loc[X.index, id_cols].reset_index(drop=True).copy()

    pred_df = ids.copy()
    pred_df["y_true"] = y.to_numpy()
    pred_df["y_pred_full"] = pred_full
    pred_df["y_pred_loocv"] = pred_loocv
    pred_df["residual_full"] = pred_df["y_true"] - pred_df["y_pred_full"]
    pred_df["residual_loocv"] = pred_df["y_true"] - pred_df["y_pred_loocv"]

    coef_df = pd.DataFrame(
        {
            "feature": ["intercept"] + X.columns.tolist(),
            "coefficient": [float(model.intercept_)] + [float(v) for v in model.coef_],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    summary = {
        "input_csv": str(in_csv),
        "target": args.y,
        "dataset_col": dataset_col,
        "region_col": region_col,
        "alpha": args.alpha,
        "n_rows": int(len(y)),
        "n_predictors": int(X.shape[1]),
        "used_numeric_predictors": used_x_cols,
        "used_model_matrix_columns": X.columns.tolist(),
        "full_fit_r2": safe_r2(y.to_numpy(), pred_full),
        "full_fit_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
        "full_fit_mae": float(mean_absolute_error(y, pred_full)),
        "loocv_r2": safe_r2(y.to_numpy(), pred_loocv),
        "loocv_rmse": float(np.sqrt(mean_squared_error(y, pred_loocv))),
        "loocv_mae": float(mean_absolute_error(y, pred_loocv)),
    }

    X_out = X.copy()
    X_out.insert(0, "y_true", y.to_numpy())

    pred_path = outdir / "region_model_predictions.csv"
    coef_path = outdir / "region_model_coefficients.csv"
    summary_path = outdir / "region_model_summary.json"
    X_path = outdir / "region_model_matrix_used.csv"

    pred_df.to_csv(pred_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    X_out.to_csv(X_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" -", pred_path)
    print(" -", coef_path)
    print(" -", X_path)
    print(" -", summary_path)
    print()
    print("Target:", args.y)
    print("Dataset column:", dataset_col)
    print("Region column:", region_col)
    print("Rows:", len(y))
    print("Predictors:", X.shape[1])
    print("Full-fit R2:", summary["full_fit_r2"])
    print("LOOCV RMSE:", summary["loocv_rmse"])
    print("LOOCV MAE:", summary["loocv_mae"])


if __name__ == "__main__":
    main()
