from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "06_design_matrices"
    / "joint_region_bridge_model_matrix_crc_breast_core4.csv"
)

OUTDIR = PROJECT_ROOT / "outputs" / "07_ridge_models"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [
    "bridge_epi_like_mean_joint_z",
    "bridge_fibroblast_mean_joint_z",
    "bridge_smooth_myoepi_mean_joint_z",
    "bridge_ECM_mean_joint_z",
]

TARGET_LABELS = {
    "bridge_epi_like_mean_joint_z": "epi_like",
    "bridge_fibroblast_mean_joint_z": "fibroblast",
    "bridge_smooth_myoepi_mean_joint_z": "smooth_myoepi",
    "bridge_ECM_mean_joint_z": "ECM",
}

ADJUSTMENT_COLS = [
    "dataset_indicator_breast",
    "region_indicator_Q2_UR",
    "region_indicator_Q3_LL",
    "region_indicator_Q4_LR",
]

ALPHAS = np.logspace(-3, 3, 49)


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def powerset(items: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    """All subsets of items, including empty and full."""
    items = list(items)
    for r in range(len(items) + 1):
        for combo in combinations(items, r):
            yield combo


def make_pipeline(alpha: float) -> Pipeline:
    """
    Ridge pipeline.

    StandardScaler is fit within each training fold to avoid leakage.
    The ridge intercept is not penalized by scikit-learn Ridge.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=float(alpha), fit_intercept=True)),
        ]
    )


def cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if not np.isclose(ss_tot, 0.0) else np.nan

    corr = np.nan
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "loocv_rmse": rmse,
        "loocv_mae": mae,
        "loocv_r2": r2,
        "loocv_pearson_r": corr,
    }


def fit_loocv(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: Sequence[str],
    alpha: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X = df[list(predictor_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    n = len(df)
    preds = np.full(n, np.nan, dtype=float)

    for holdout_idx in range(n):
        train_idx = np.array([i for i in range(n) if i != holdout_idx], dtype=int)

        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[[holdout_idx], :]

        model = make_pipeline(alpha)
        model.fit(X_train, y_train)
        preds[holdout_idx] = float(model.predict(X_test)[0])

    metrics = cv_metrics(y, preds)

    pred_df = pd.DataFrame(
        {
            "target": TARGET_LABELS[target_col],
            "target_col": target_col,
            "observation_id": df["observation_id"].astype(str).values,
            "dataset": df["dataset"].astype(str).values,
            "region_id": df["region_id"].astype(str).values,
            "alpha": float(alpha),
            "y_true": y,
            "y_pred": preds,
            "residual": y - preds,
        }
    )

    return metrics, pred_df


def refit_full_data(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X = df[list(predictor_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    model = make_pipeline(alpha)
    model.fit(X, y)

    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]

    rows = []

    rows.append(
        {
            "target": TARGET_LABELS[target_col],
            "target_col": target_col,
            "term": "intercept",
            "coefficient": float(ridge.intercept_),
            "coefficient_scale": "standardized_predictor_scale",
            "alpha": float(alpha),
            "predictor_mean_used_for_scaling": np.nan,
            "predictor_scale_used_for_scaling": np.nan,
        }
    )

    for term, coef, mean, scale in zip(
        predictor_cols,
        ridge.coef_,
        scaler.mean_,
        scaler.scale_,
    ):
        rows.append(
            {
                "target": TARGET_LABELS[target_col],
                "target_col": target_col,
                "term": term,
                "coefficient": float(coef),
                "coefficient_scale": "standardized_predictor_scale",
                "alpha": float(alpha),
                "predictor_mean_used_for_scaling": float(mean),
                "predictor_scale_used_for_scaling": float(scale),
            }
        )

    return pd.DataFrame(rows)


def evaluate_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_label = TARGET_LABELS[target_col]

    bridge_predictor_pool = [c for c in TARGET_COLS if c != target_col]

    candidate_rows = []
    candidate_pred_tables = {}

    for bridge_subset in powerset(bridge_predictor_pool):
        predictor_cols = ADJUSTMENT_COLS + list(bridge_subset)

        model_class = "adjustment_only" if len(bridge_subset) == 0 else "bridge_panel"

        for alpha in ALPHAS:
            metrics, pred_df = fit_loocv(
                df=df,
                target_col=target_col,
                predictor_cols=predictor_cols,
                alpha=float(alpha),
            )

            candidate_id = (
                f"{target_label}__k{len(bridge_subset)}__"
                f"alpha_{alpha:.6g}__"
                f"{'none' if len(bridge_subset) == 0 else '+'.join(bridge_subset)}"
            )

            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": target_label,
                    "target_col": target_col,
                    "model_class": model_class,
                    "alpha": float(alpha),
                    "n_observations": int(len(df)),
                    "n_adjustment_terms": int(len(ADJUSTMENT_COLS)),
                    "n_bridge_predictors": int(len(bridge_subset)),
                    "n_total_predictors": int(len(predictor_cols)),
                    "adjustment_terms": ";".join(ADJUSTMENT_COLS),
                    "bridge_predictors": ";".join(bridge_subset),
                    "predictor_terms": ";".join(predictor_cols),
                    **metrics,
                }
            )

            candidate_pred_tables[candidate_id] = pred_df.assign(
                candidate_id=candidate_id,
                model_class=model_class,
                n_bridge_predictors=int(len(bridge_subset)),
                predictor_terms=";".join(predictor_cols),
            )

    all_candidates = pd.DataFrame(candidate_rows)

    # Selection rule:
    #   1. lowest LOOCV RMSE
    #   2. if tied, fewer bridge predictors
    #   3. if tied, smaller alpha
    # This keeps selected panels compact and conservative.
    selected = (
        all_candidates.sort_values(
            ["loocv_rmse", "n_bridge_predictors", "alpha"],
            ascending=[True, True, True],
        )
        .iloc[0]
        .copy()
    )

    selected_candidate_id = str(selected["candidate_id"])
    selected_pred = candidate_pred_tables[selected_candidate_id].copy()

    selected_predictors = str(selected["predictor_terms"]).split(";")
    selected_alpha = float(selected["alpha"])

    selected_coef = refit_full_data(
        df=df,
        target_col=target_col,
        predictor_cols=selected_predictors,
        alpha=selected_alpha,
    ).assign(
        candidate_id=selected_candidate_id,
        model_class=selected["model_class"],
        n_bridge_predictors=int(selected["n_bridge_predictors"]),
        predictor_terms=selected["predictor_terms"],
    )

    selected_summary = pd.DataFrame([selected])

    return all_candidates, selected_summary, selected_coef, selected_pred


def write_readme(outpath: Path) -> None:
    lines = []

    lines.append("Region-aware ridge modeling outputs")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Scope")
    lines.append("-" * 70)
    lines.append(
        "These outputs summarize regularized exploratory association models "
        "fit to the clean eight-row CRC--Breast region-aware bridge design matrix."
    )
    lines.append("")
    lines.append("Important interpretation")
    lines.append("-" * 70)
    lines.append(
        "Because the design matrix contains only eight region-level observations, "
        "these ridge models should not be interpreted as confirmatory prediction, "
        "causal inference, or population-level dependency estimation. They are used "
        "only to summarize candidate relationships among bridge features after "
        "adjusting for dataset and coarse region."
    )
    lines.append("")
    lines.append("Model structure")
    lines.append("-" * 70)
    lines.append(
        "For each target bridge feature, every candidate model retained the dataset "
        "indicator and three region indicators. Candidate bridge panels were then "
        "formed from all subsets of the remaining three non-target bridge features."
    )
    lines.append("")
    lines.append("Model selection")
    lines.append("-" * 70)
    lines.append(
        "Models were evaluated by leave-one-out cross-validation over the eight "
        "region-level observations. The selected model for each target minimized "
        "LOOCV RMSE. Ties were resolved by choosing fewer bridge predictors and then "
        "smaller alpha."
    )
    lines.append("")
    lines.append("Regularization")
    lines.append("-" * 70)
    lines.append(
        "Ridge alpha values were evaluated on a log-spaced grid from 1e-3 to 1e3. "
        "Predictors were standardized within each training fold using StandardScaler. "
        "The final selected model coefficients were refit on all eight observations "
        "and are reported on the standardized-predictor scale."
    )
    lines.append("")

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input model matrix: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = [
        "observation_id",
        "dataset",
        "region_id",
        *ADJUSTMENT_COLS,
        *TARGET_COLS,
    ]
    require_columns(df, required)

    all_candidate_parts = []
    selected_summary_parts = []
    selected_coef_parts = []
    selected_pred_parts = []

    for target_col in TARGET_COLS:
        all_candidates, selected_summary, selected_coef, selected_pred = evaluate_target(
            df=df,
            target_col=target_col,
        )

        all_candidate_parts.append(all_candidates)
        selected_summary_parts.append(selected_summary)
        selected_coef_parts.append(selected_coef)
        selected_pred_parts.append(selected_pred)

        selected = selected_summary.iloc[0]
        print(
            f"[OK] target={selected['target']}: "
            f"selected RMSE={selected['loocv_rmse']:.4f}, "
            f"R2={selected['loocv_r2']:.4f}, "
            f"alpha={selected['alpha']:.6g}, "
            f"bridge_predictors={selected['bridge_predictors'] or 'none'}"
        )

    all_candidates_df = pd.concat(all_candidate_parts, ignore_index=True)
    selected_summary_df = pd.concat(selected_summary_parts, ignore_index=True)
    selected_coef_df = pd.concat(selected_coef_parts, ignore_index=True)
    selected_pred_df = pd.concat(selected_pred_parts, ignore_index=True)

    all_candidates_out = OUTDIR / "ridge_all_candidate_models.csv"
    selected_summary_out = OUTDIR / "ridge_model_selection_summary.csv"
    selected_coef_out = OUTDIR / "ridge_coefficients_selected_models.csv"
    selected_pred_out = OUTDIR / "ridge_loocv_predictions_selected_models.csv"
    readme_out = OUTDIR / "ridge_modeling_readme.txt"

    all_candidates_df.to_csv(all_candidates_out, index=False)
    selected_summary_df.to_csv(selected_summary_out, index=False)
    selected_coef_df.to_csv(selected_coef_out, index=False)
    selected_pred_df.to_csv(selected_pred_out, index=False)
    write_readme(readme_out)

    print("")
    print(f"[OK] wrote {all_candidates_out}")
    print(f"[OK] wrote {selected_summary_out}")
    print(f"[OK] wrote {selected_coef_out}")
    print(f"[OK] wrote {selected_pred_out}")
    print(f"[OK] wrote {readme_out}")


if __name__ == "__main__":
    main()
