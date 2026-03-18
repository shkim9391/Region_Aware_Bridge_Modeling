from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr


SELECTED_PANEL: Dict[str, Dict[str, List[str] | str]] = {
    "epi": {
        "y": "bridge_epi_like_mean_z",
        "x": [
            "bridge_ECM_mean_z",
            "bridge_smooth_myoepi_mean_z",
        ],
    },
    "fib": {
        "y": "bridge_fibroblast_mean_z",
        "x": [
            "bridge_ECM_mean_z",
            "bridge_epi_like_mean_z",
        ],
    },
    "smy": {
        "y": "bridge_smooth_myoepi_mean_z",
        "x": [
            "bridge_ECM_mean_z",
            "bridge_epi_like_mean_z",
        ],
    },
    "ecm": {
        "y": "bridge_ECM_mean_z",
        "x": [
            "bridge_smooth_myoepi_mean_z",
        ],
    },
}


def encode_levels(values: pd.Series) -> Tuple[np.ndarray, List[str]]:
    levels = list(pd.unique(values.astype(str)))
    mapping = {k: i for i, k in enumerate(levels)}
    idx = values.astype(str).map(mapping).to_numpy(dtype=int)
    return idx, levels


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_clean_df(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    dataset_col: str,
    region_col: str,
    id_cols: List[str],
) -> pd.DataFrame:
    # Deduplicate requested columns while preserving order
    needed = list(dict.fromkeys([y_col] + x_cols + [dataset_col, region_col] + id_cols))

    require_columns(df, [y_col] + x_cols + [dataset_col, region_col])

    out = df[needed].copy()
    
    print("Columns in clean frame:", list(out.columns))
    print("Duplicate columns:", out.columns[out.columns.duplicated()].tolist())

    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    for c in x_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Build mask as a clean boolean Series aligned to rows
    keep = pd.Series(True, index=out.index, dtype=bool)
    keep = keep & out[y_col].notna()

    for c in x_cols:
        keep = keep & out[c].notna()

    keep = keep & out[dataset_col].notna()
    keep = keep & out[region_col].notna()

    out = out.loc[keep].reset_index(drop=True)
    if out.empty:
        raise ValueError("No rows remain after dropping missing values.")

    return out


def summarize_scalar(
    idata: az.InferenceData,
    var_names: List[str],
    hdi_prob: float,
) -> pd.DataFrame:
    return az.summary(
        idata,
        var_names=var_names,
        hdi_prob=hdi_prob,
        round_to=None,
    ).reset_index(names="parameter")


def dataarray_mean_hdi(da, hdi_prob: float):
    mean = da.mean(dim=("chain", "draw")).to_numpy()
    hdi = az.hdi(da, hdi_prob=hdi_prob)

    # ArviZ/xarray may return either a DataArray or a Dataset here.
    if isinstance(hdi, xr.Dataset):
        var_name = da.name if da.name in hdi.data_vars else list(hdi.data_vars)[0]
        hdi_da = hdi[var_name]
    else:
        hdi_da = hdi

    lower = hdi_da.sel(hdi="lower").to_numpy()
    upper = hdi_da.sel(hdi="higher").to_numpy()
    return mean, lower, upper


def ic_value(obj, *candidate_names: str) -> float:
    # Works for ArviZ ELPDData / pandas-like returns across naming changes.
    for name in candidate_names:
        if hasattr(obj, name):
            try:
                return float(getattr(obj, name))
            except Exception:
                pass
        try:
            if hasattr(obj, "index") and name in obj.index:
                return float(obj[name])
        except Exception:
            pass
        try:
            if name in obj:
                return float(obj[name])
        except Exception:
            pass

    available = []
    try:
        available = list(obj.index)
    except Exception:
        pass
    raise KeyError(f"None of {candidate_names} found. Available keys: {available}")
    
    
def ic_value_or_nan(obj, *candidate_names: str) -> float:
    for name in candidate_names:
        if hasattr(obj, name):
            try:
                return float(getattr(obj, name))
            except Exception:
                pass
        try:
            if hasattr(obj, "index") and name in obj.index:
                return float(obj[name])
        except Exception:
            pass
        try:
            if name in obj:
                return float(obj[name])
        except Exception:
            pass
    return float("nan")   


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit a small Bayesian hierarchical region-aware bridge model with selected best-panel formulas."
    )
    ap.add_argument(
        "--in",
        dest="in_csv",
        default="output/output_crc_breast_bridge_compare_core4_regions4/ou_branching_bridge_design_matrix_crc_breast_core4_regions4.csv",
        help="Input joint design matrix CSV",
    )
    ap.add_argument(
        "--target-label",
        choices=sorted(SELECTED_PANEL.keys()),
        required=True,
        help="Which selected best-panel target to fit: epi, fib, smy, or ecm",
    )
    ap.add_argument(
        "--dataset-col",
        default="bridge_dataset",
        help="Dataset/cohort column",
    )
    ap.add_argument(
        "--region-col",
        default="region_id",
        help="Region column",
    )
    ap.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Posterior draws per chain",
    )
    ap.add_argument(
        "--tune",
        type=int,
        default=2000,
        help="Tuning steps per chain",
    )
    ap.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains",
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=4,
        help="Number of cores",
    )
    ap.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
        help="NUTS target_accept",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed",
    )
    ap.add_argument(
        "--hdi-prob",
        type=float,
        default=0.94,
        help="HDI probability level",
    )
    ap.add_argument(
        "--beta-prior-scale",
        type=float,
        default=0.75,
        help="Prior SD for numeric coefficients",
    )
    ap.add_argument(
        "--dataset-prior-scale",
        type=float,
        default=0.50,
        help="HalfNormal prior scale for dataset random-effect SD",
    )
    ap.add_argument(
        "--region-prior-scale",
        type=float,
        default=0.75,
        help="HalfNormal prior scale for region random-effect SD",
    )
    ap.add_argument(
        "--sigma-prior-scale",
        type=float,
        default=1.0,
        help="HalfNormal prior scale for residual SD",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory; default uses target label",
    )
    args = ap.parse_args()

    spec = SELECTED_PANEL[args.target_label]
    y_col = str(spec["y"])
    x_cols = list(spec["x"])

    in_csv = Path(args.in_csv)
    outdir = Path(
        args.outdir
        or f"output/output_crc_breast_bridge_compare_core4_regions4/region_model_bayes_{args.target_label}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    id_cols = [c for c in ["bridge_dataset", "dataset", "sample_id", "region_id", "region_label"] if c in df.columns]
    clean = build_clean_df(
        df=df,
        y_col=y_col,
        x_cols=x_cols,
        dataset_col=args.dataset_col,
        region_col=args.region_col,
        id_cols=id_cols,
    )

    y = clean[y_col].to_numpy(dtype=float)
    X = clean[x_cols].to_numpy(dtype=float)

    dataset_idx, dataset_levels = encode_levels(clean[args.dataset_col])
    region_idx, region_levels = encode_levels(clean[args.region_col])

    coords = {
        "obs_id": np.arange(len(clean)),
        "predictor": x_cols,
        "dataset": dataset_levels,
        "region": region_levels,
    }

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("x_data", X, dims=("obs_id", "predictor"))
        y_data = pm.Data("y_data", y, dims="obs_id")
        dataset_idx_data = pm.Data("dataset_idx", dataset_idx, dims="obs_id")
        region_idx_data = pm.Data("region_idx", region_idx, dims="obs_id")

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=args.beta_prior_scale,
            dims="predictor",
        )

        dataset_sigma = pm.HalfNormal(
            "dataset_sigma",
            sigma=args.dataset_prior_scale,
        )
        dataset_offset = pm.Normal(
            "dataset_offset",
            mu=0.0,
            sigma=1.0,
            dims="dataset",
        )
        dataset_effect = pm.Deterministic(
            "dataset_effect",
            dataset_sigma * dataset_offset,
            dims="dataset",
        )

        region_sigma = pm.HalfNormal(
            "region_sigma",
            sigma=args.region_prior_scale,
        )
        region_offset = pm.Normal(
            "region_offset",
            mu=0.0,
            sigma=1.0,
            dims="region",
        )
        region_effect = pm.Deterministic(
            "region_effect",
            region_sigma * region_offset,
            dims="region",
        )

        sigma = pm.HalfNormal(
            "sigma",
            sigma=args.sigma_prior_scale,
        )

        mu = pm.Deterministic(
            "mu",
            alpha
            + dataset_effect[dataset_idx_data]
            + region_effect[region_idx_data]
            + pm.math.sum(x_data * beta, axis=1),
            dims="obs_id",
        )

        pm.Normal(
            "y_like",
            mu=mu,
            sigma=sigma,
            observed=y_data,
            dims="obs_id",
        )

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=True,
        )

        idata = pm.sample_posterior_predictive(
            idata,
            var_names=["y_like"],
            extend_inferencedata=True,
            progressbar=True,
        )

    # Save full inference data
    idata_path = outdir / "idata.nc"
    idata.to_netcdf(idata_path)

    # Parameter summaries
    summary_df = summarize_scalar(
        idata=idata,
        var_names=[
            "alpha",
            "beta",
            "dataset_sigma",
            "dataset_effect",
            "region_sigma",
            "region_effect",
            "sigma",
        ],
        hdi_prob=args.hdi_prob,
    )
    summary_csv = outdir / "bayes_parameter_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Row-level predictions
    mu_da = idata.posterior["mu"]
    ypp_da = idata.posterior_predictive["y_like"]
    
    mu_mean, mu_lower, mu_upper = dataarray_mean_hdi(mu_da, args.hdi_prob)
    ypp_mean, ypp_lower, ypp_upper = dataarray_mean_hdi(ypp_da, args.hdi_prob)

    pred_df = clean[id_cols].copy()
    pred_df["y_true"] = y
    pred_df["mu_mean"] = mu_mean
    pred_df["mu_hdi_lower"] = mu_lower
    pred_df["mu_hdi_upper"] = mu_upper
    pred_df["y_pp_mean"] = ypp_mean
    pred_df["y_pp_hdi_lower"] = ypp_lower
    pred_df["y_pp_hdi_upper"] = ypp_upper
    pred_df["residual_mu_mean"] = pred_df["y_true"] - pred_df["mu_mean"]

    pred_csv = outdir / "bayes_row_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    # A compact coefficient table
    beta_summary = summary_df.loc[summary_df["parameter"].str.startswith("beta[")].copy()
    beta_summary["predictor"] = beta_summary["parameter"].str.extract(r"beta\[(.*)\]", expand=False)
    beta_csv = outdir / "bayes_beta_summary.csv"
    beta_summary.to_csv(beta_csv, index=False)

    dataset_summary = summary_df.loc[summary_df["parameter"].str.startswith("dataset_effect[")].copy()
    region_summary = summary_df.loc[summary_df["parameter"].str.startswith("region_effect[")].copy()
    dataset_csv = outdir / "bayes_dataset_effect_summary.csv"
    region_csv = outdir / "bayes_region_effect_summary.csv"
    dataset_summary.to_csv(dataset_csv, index=False)
    region_summary.to_csv(region_csv, index=False)

    # Diagnostics / model summary
    loo_obj = az.loo(idata, pointwise=False)

    try:
        waic_obj = az.waic(idata, pointwise=False)
        waic_keys = list(getattr(waic_obj, "index", []))
        waic_error = None
    except Exception as e:
        waic_obj = None
        waic_keys = []
        waic_error = repr(e)
    
    print("LOO object:")
    print(loo_obj)
    print("LOO keys:", list(getattr(loo_obj, "index", [])))
    print("WAIC keys:", waic_keys)
    if waic_error is not None:
        print("WAIC error:", waic_error)

    div_total = int(idata.sample_stats["diverging"].sum().to_numpy())
    max_rhat = float(summary_df["r_hat"].dropna().max()) if "r_hat" in summary_df.columns else float("nan")
    min_ess_bulk = float(summary_df["ess_bulk"].dropna().min()) if "ess_bulk" in summary_df.columns else float("nan")

    out_summary = {
        "input_csv": str(in_csv),
        "target_label": args.target_label,
        "target_column": y_col,
        "predictor_columns": x_cols,
        "dataset_col": args.dataset_col,
        "region_col": args.region_col,
        "n_rows": int(len(clean)),
        "n_predictors": int(len(x_cols)),
        "dataset_levels": dataset_levels,
        "region_levels": region_levels,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_seed": args.random_seed,
        "beta_prior_scale": args.beta_prior_scale,
        "dataset_prior_scale": args.dataset_prior_scale,
        "region_prior_scale": args.region_prior_scale,
        "sigma_prior_scale": args.sigma_prior_scale,
        "divergences": div_total,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "loo_elpd": ic_value_or_nan(loo_obj, "elpd_loo", "loo", "elpd"),
        "loo_p": ic_value_or_nan(loo_obj, "p_loo", "p_ic", "pIC"),
        "waic_elpd": ic_value_or_nan(waic_obj, "elpd_waic", "waic", "elpd"),
        "waic_p": ic_value_or_nan(waic_obj, "p_waic", "p_ic", "pIC"),
        "waic_error": waic_error,
    }

    summary_json = outdir / "bayes_model_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)

    print("Saved:")
    print(" -", idata_path)
    print(" -", summary_csv)
    print(" -", beta_csv)
    print(" -", dataset_csv)
    print(" -", region_csv)
    print(" -", pred_csv)
    print(" -", summary_json)
    print()
    print("Target:", args.target_label, "->", y_col)
    print("Predictors:", ", ".join(x_cols))
    print("Rows:", len(clean))
    print("Divergences:", div_total)
    print("Max r_hat:", max_rhat)
    print("Min ess_bulk:", min_ess_bulk)
    print("LOO elpd:", ic_value_or_nan(loo_obj, "elpd_loo", "loo", "elpd"))
    print("WAIC elpd:", ic_value_or_nan(waic_obj, "elpd_waic", "waic", "elpd"))


if __name__ == "__main__":
    main()
