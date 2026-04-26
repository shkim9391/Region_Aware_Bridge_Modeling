from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_MATRIX_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "06_design_matrices"
    / "joint_region_bridge_model_matrix_crc_breast_core4.csv"
)

RIDGE_SELECTION_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "07_ridge_models"
    / "ridge_model_selection_summary.csv"
)

OUTDIR = PROJECT_ROOT / "outputs" / "08_bayesian_models"
OUTDIR.mkdir(parents=True, exist_ok=True)

ADJUSTMENT_COLS = [
    "dataset_indicator_breast",
    "region_indicator_Q2_UR",
    "region_indicator_Q3_LL",
    "region_indicator_Q4_LR",
]

TARGET_TO_COL = {
    "epi_like": "bridge_epi_like_mean_joint_z",
    "fibroblast": "bridge_fibroblast_mean_joint_z",
    "smooth_myoepi": "bridge_smooth_myoepi_mean_joint_z",
    "ECM": "bridge_ECM_mean_joint_z",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Bayesian models for ridge-selected region-aware bridge panels."
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning draws per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains.")
    parser.add_argument("--target-accept", type=float, default=0.95, help="NUTS target_accept.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--hdi-prob",
        type=float,
        default=0.94,
        help="Highest-density interval probability.",
    )
    parser.add_argument(
        "--prior-scale-coef",
        type=float,
        default=1.0,
        help="Normal prior SD for coefficients.",
    )
    parser.add_argument(
        "--prior-scale-intercept",
        type=float,
        default=1.0,
        help="Normal prior SD for intercept.",
    )
    parser.add_argument(
        "--prior-scale-sigma",
        type=float,
        default=1.0,
        help="HalfNormal prior scale for residual sigma.",
    )
    return parser.parse_args()


def split_predictor_terms(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "":
        return []
    return [x for x in text.split(";") if x]


def require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def fit_one_model(
    df: pd.DataFrame,
    target: str,
    target_col: str,
    predictor_terms: List[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X = df[predictor_terms].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    coords = {
        "observation": df["observation_id"].astype(str).tolist(),
        "term": predictor_terms,
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("observation", "term"))
        y_data = pm.Data("y", y, dims="observation")

        intercept = pm.Normal(
            "intercept",
            mu=0.0,
            sigma=args.prior_scale_intercept,
        )
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=args.prior_scale_coef,
            dims="term",
        )
        sigma = pm.HalfNormal(
            "sigma",
            sigma=args.prior_scale_sigma,
        )

        mu = pm.Deterministic(
            "mu",
            intercept + pm.math.dot(X_data, beta),
            dims="observation",
        )

        pm.Normal(
            "likelihood",
            mu=mu,
            sigma=sigma,
            observed=y_data,
            dims="observation",
        )

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True,
        )

        posterior_predictive = pm.sample_posterior_predictive(
            idata,
            var_names=["likelihood"],
            random_seed=args.seed,
            progressbar=True,
        )

    # Posterior summary for parameters.
    summary = az.summary(
        idata,
        var_names=["intercept", "beta", "sigma"],
        hdi_prob=args.hdi_prob,
        round_to=None,
    ).reset_index(names="parameter")

    summary.insert(0, "target", target)
    summary.insert(1, "target_col", target_col)

    # Clean term labels for beta rows.
    summary["term"] = summary["parameter"]
    for term in predictor_terms:
        summary.loc[summary["parameter"] == f"beta[{term}]", "term"] = term
    summary.loc[summary["parameter"] == "intercept", "term"] = "intercept"
    summary.loc[summary["parameter"] == "sigma", "term"] = "sigma"

    # Diagnostics subset.
    diag_cols = [
        c
        for c in [
            "target",
            "target_col",
            "parameter",
            "term",
            "r_hat",
            "ess_bulk",
            "ess_tail",
            "mcse_mean",
            "mcse_sd",
        ]
        if c in summary.columns
    ]
    diagnostics = summary[diag_cols].copy()

    # Posterior predictive summary.
    # Shape is usually chain x draw x observation.
    ppc = posterior_predictive.posterior_predictive["likelihood"].values
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])

    ppc_rows = []
    for i, obs_id in enumerate(df["observation_id"].astype(str).tolist()):
        vals = ppc_flat[:, i]
        ppc_rows.append(
            {
                "target": target,
                "target_col": target_col,
                "observation_id": obs_id,
                "dataset": df["dataset"].iloc[i],
                "region_id": df["region_id"].iloc[i],
                "y_true": float(y[i]),
                "posterior_predictive_mean": float(np.mean(vals)),
                "posterior_predictive_sd": float(np.std(vals, ddof=0)),
                "posterior_predictive_hdi_lower": float(
                    np.quantile(vals, (1.0 - args.hdi_prob) / 2.0)
                ),
                "posterior_predictive_hdi_upper": float(
                    np.quantile(vals, 1.0 - (1.0 - args.hdi_prob) / 2.0)
                ),
            }
        )
    ppc_summary = pd.DataFrame(ppc_rows)

    config = {
        "target": target,
        "target_col": target_col,
        "n_observations": int(len(df)),
        "predictor_terms": ";".join(predictor_terms),
        "n_predictors": int(len(predictor_terms)),
        "draws": int(args.draws),
        "tune": int(args.tune),
        "chains": int(args.chains),
        "target_accept": float(args.target_accept),
        "seed": int(args.seed),
        "hdi_prob": float(args.hdi_prob),
        "prior_intercept": f"Normal(0, {args.prior_scale_intercept})",
        "prior_beta": f"Normal(0, {args.prior_scale_coef})",
        "prior_sigma": f"HalfNormal({args.prior_scale_sigma})",
        "sampler": "PyMC NUTS",
        "pymc_version": pm.__version__,
        "arviz_version": az.__version__,
    }

    return summary, diagnostics, ppc_summary, config


def write_readme(outpath: Path, args: argparse.Namespace) -> None:
    lines = []

    lines.append("Region-aware Bayesian modeling outputs")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Scope")
    lines.append("-" * 70)
    lines.append(
        "These outputs summarize Bayesian Gaussian regression models fit to the "
        "ridge-selected exploratory bridge panels."
    )
    lines.append("")
    lines.append("Important interpretation")
    lines.append("-" * 70)
    lines.append(
        "The joint CRC--Breast design matrix contains only eight region-level "
        "observations. Therefore, these Bayesian models should be interpreted as "
        "uncertainty-aware exploratory summaries, not confirmatory population-level "
        "inference."
    )
    lines.append("")
    lines.append("Model structure")
    lines.append("-" * 70)
    lines.append(
        "For each target bridge feature, the Bayesian model uses the same predictor "
        "terms selected by the ridge-modeling step. These always include dataset "
        "and region indicators, with ridge-selected non-target bridge predictors "
        "when selected."
    )
    lines.append("")
    lines.append("Likelihood and priors")
    lines.append("-" * 70)
    lines.append("y_r ~ Normal(mu_r, sigma)")
    lines.append("mu_r = intercept + X_r beta")
    lines.append(f"intercept ~ Normal(0, {args.prior_scale_intercept})")
    lines.append(f"beta_j ~ Normal(0, {args.prior_scale_coef})")
    lines.append(f"sigma ~ HalfNormal({args.prior_scale_sigma})")
    lines.append("")
    lines.append("Sampling")
    lines.append("-" * 70)
    lines.append("Sampler: PyMC NUTS")
    lines.append(f"chains: {args.chains}")
    lines.append(f"tune: {args.tune}")
    lines.append(f"draws: {args.draws}")
    lines.append(f"target_accept: {args.target_accept}")
    lines.append(f"hdi_prob: {args.hdi_prob}")
    lines.append("")

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if not MODEL_MATRIX_CSV.exists():
        raise FileNotFoundError(f"Missing model matrix: {MODEL_MATRIX_CSV}")
    if not RIDGE_SELECTION_CSV.exists():
        raise FileNotFoundError(f"Missing ridge selection summary: {RIDGE_SELECTION_CSV}")

    model_df = pd.read_csv(MODEL_MATRIX_CSV)
    ridge_df = pd.read_csv(RIDGE_SELECTION_CSV)

    require_columns(
        model_df,
        ["observation_id", "dataset", "region_id", *ADJUSTMENT_COLS, *TARGET_TO_COL.values()],
        "model matrix",
    )
    require_columns(
        ridge_df,
        ["target", "target_col", "predictor_terms"],
        "ridge selection summary",
    )

    posterior_parts = []
    diagnostic_parts = []
    ppc_parts = []
    config_rows = []

    for _, row in ridge_df.iterrows():
        target = str(row["target"])
        target_col = str(row["target_col"])

        if target not in TARGET_TO_COL:
            raise ValueError(f"Unknown target in ridge selection: {target}")

        predictor_terms = split_predictor_terms(row["predictor_terms"])

        if not predictor_terms:
            # Should not happen because adjustment terms are always retained,
            # but keep a defensive check.
            predictor_terms = ADJUSTMENT_COLS

        require_columns(model_df, [target_col, *predictor_terms], f"Bayesian model for {target}")

        print("")
        print("=" * 70)
        print(f"Fitting Bayesian model for target={target}")
        print(f"Target column: {target_col}")
        print(f"Predictors: {', '.join(predictor_terms)}")
        print("=" * 70)

        posterior_summary, diagnostics, ppc_summary, config = fit_one_model(
            df=model_df,
            target=target,
            target_col=target_col,
            predictor_terms=predictor_terms,
            args=args,
        )

        posterior_parts.append(posterior_summary)
        diagnostic_parts.append(diagnostics)
        ppc_parts.append(ppc_summary)
        config_rows.append(config)

        # Small console diagnostic.
        max_rhat = diagnostics["r_hat"].dropna().max() if "r_hat" in diagnostics.columns else np.nan
        min_ess = diagnostics["ess_bulk"].dropna().min() if "ess_bulk" in diagnostics.columns else np.nan
        print(f"[OK] target={target}: max_r_hat={max_rhat}, min_ess_bulk={min_ess}")

    posterior_df = pd.concat(posterior_parts, ignore_index=True)
    diagnostics_df = pd.concat(diagnostic_parts, ignore_index=True)
    ppc_df = pd.concat(ppc_parts, ignore_index=True)
    config_df = pd.DataFrame(config_rows)

    posterior_out = OUTDIR / "bayesian_posterior_summary.csv"
    diagnostics_out = OUTDIR / "bayesian_diagnostics.csv"
    ppc_out = OUTDIR / "bayesian_posterior_predictive_summary.csv"
    config_out = OUTDIR / "bayesian_model_config.csv"
    readme_out = OUTDIR / "bayesian_modeling_readme.txt"

    posterior_df.to_csv(posterior_out, index=False)
    diagnostics_df.to_csv(diagnostics_out, index=False)
    ppc_df.to_csv(ppc_out, index=False)
    config_df.to_csv(config_out, index=False)
    write_readme(readme_out, args)

    print("")
    print(f"[OK] wrote {posterior_out}")
    print(f"[OK] wrote {diagnostics_out}")
    print(f"[OK] wrote {ppc_out}")
    print(f"[OK] wrote {config_out}")
    print(f"[OK] wrote {readme_out}")


if __name__ == "__main__":
    main()
