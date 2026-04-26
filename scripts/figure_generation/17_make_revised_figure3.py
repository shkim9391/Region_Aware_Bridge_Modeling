from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "07_ridge_models"
    / "ridge_model_selection_summary.csv"
)

COEF_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "07_ridge_models"
    / "ridge_coefficients_selected_models.csv"
)

OUTDIR = PROJECT_ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Figure3_revised_ridge_exploratory_modeling.png"
OUT_PDF = OUTDIR / "Figure3_revised_ridge_exploratory_modeling.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

TARGET_ORDER = ["epi_like", "fibroblast", "smooth_myoepi", "ECM"]

TARGET_DISPLAY = {
    "epi_like": "Epithelial-like",
    "fibroblast": "Fibroblast",
    "smooth_myoepi": "Smooth/\nMyoepithelial",
    "ECM": "ECM",
}

TARGET_SHORT = {
    "epi_like": "Epi-like",
    "fibroblast": "Fibroblast",
    "smooth_myoepi": "Smooth/Myo",
    "ECM": "ECM",
}

TERM_DISPLAY = {
    "dataset_indicator_breast": "Breast indicator",
    "region_indicator_Q2_UR": "Q2_UR",
    "region_indicator_Q3_LL": "Q3_LL",
    "region_indicator_Q4_LR": "Q4_LR",
    "bridge_epi_like_mean_joint_z": "Epithelial-like",
    "bridge_fibroblast_mean_joint_z": "Fibroblast",
    "bridge_smooth_myoepi_mean_joint_z": "Smooth/Myoepithelial",
    "bridge_ECM_mean_joint_z": "ECM",
}

PREDICTOR_ORDER = [
    "dataset_indicator_breast",
    "region_indicator_Q2_UR",
    "region_indicator_Q3_LL",
    "region_indicator_Q4_LR",
    "bridge_epi_like_mean_joint_z",
    "bridge_fibroblast_mean_joint_z",
    "bridge_smooth_myoepi_mean_joint_z",
    "bridge_ECM_mean_joint_z",
]

COLORS = {
    "crc": "#1f77b4",
    "breast": "#ff7f0e",
    "bar": "#4f79b8",
    "selected": "#74c476",
    "not_selected": "#f1f1f1",
    "positive": "#4f79b8",
    "negative": "#d95f02",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="right",
        va="top",
    )


def parse_terms(text) -> list[str]:
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [x for x in text.split(";") if x]


def ordered_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["target"] = pd.Categorical(out["target"], categories=TARGET_ORDER, ordered=True)
    out = out.sort_values("target").reset_index(drop=True)
    out["target"] = out["target"].astype(str)
    return out


def clean_term_label(term: str) -> str:
    return TERM_DISPLAY.get(term, term)


def make_selection_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        target = str(row["target"])
        selected_terms = set(parse_terms(row["predictor_terms"]))

        for term in PREDICTOR_ORDER:
            if term == str(row["target_col"]):
                # Do not allow a target to predict itself.
                selected = np.nan
            else:
                selected = 1 if term in selected_terms else 0

            rows.append({
                "target": target,
                "term": term,
                "selected": selected,
            })

    mat = (
        pd.DataFrame(rows)
        .pivot(index="term", columns="target", values="selected")
        .reindex(index=PREDICTOR_ORDER, columns=TARGET_ORDER)
    )
    return mat


def panel_a_performance(ax, summary_df: pd.DataFrame) -> None:
    x = np.arange(len(summary_df))
    rmse = summary_df["loocv_rmse"].to_numpy(dtype=float)
    r2 = summary_df["loocv_r2"].to_numpy(dtype=float)
    alpha = summary_df["alpha"].to_numpy(dtype=float)
    n_bridge = summary_df["n_bridge_predictors"].to_numpy(dtype=int)

    bars = ax.bar(x, rmse, color=COLORS["bar"], edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_SHORT[t] for t in summary_df["target"]], rotation=0)
    ax.set_ylabel("LOOCV RMSE")
    ax.set_title("Selected ridge-model performance", pad=8)

    ymax = float(np.max(rmse)) * 1.32
    ax.set_ylim(0, ymax)

    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + ymax * 0.025,
            f"R²={r2[i]:.2f}\nα={alpha[i]:.3g}\nk={n_bridge[i]}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.text(
        0.02,
        0.96,
        "LOOCV over 8 median-quadrant observations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#dddddd"),
    )

    add_panel_label(ax, "A")


def panel_b_selection(ax, summary_df: pd.DataFrame) -> None:
    mat = make_selection_matrix(summary_df)

    plot_mat = mat.copy()
    plot_mat = plot_mat.fillna(-1)

    # -1 self-target/not applicable = light hatch-like gray
    # 0 not selected = white
    # 1 selected = green
    cmap = ListedColormap(["#dddddd", COLORS["not_selected"], COLORS["selected"]])

    # remap -1,0,1 to 0,1,2 for cmap
    display = plot_mat.replace({-1: 0, 0: 1, 1: 2}).to_numpy(dtype=float)

    ax.imshow(display, aspect="auto", cmap=cmap, vmin=0, vmax=2)

    ax.set_yticks(np.arange(len(PREDICTOR_ORDER)))
    ax.set_yticklabels([clean_term_label(t) for t in PREDICTOR_ORDER])

    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_SHORT[t] for t in TARGET_ORDER], rotation=0)

    ax.set_title("Selected predictors in ridge models", pad=8)

    for i, term in enumerate(PREDICTOR_ORDER):
        for j, target in enumerate(TARGET_ORDER):
            val = mat.loc[term, target]
            if pd.isna(val):
                txt = "—"
                color = "#777777"
            elif int(val) == 1:
                txt = "●"
                color = "white"
            else:
                txt = "·"
                color = "#999999"

            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(TARGET_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PREDICTOR_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    add_panel_label(ax, "B")


def coefficient_panel(ax, coef_df: pd.DataFrame, target: str, label: str) -> None:
    sub = coef_df[
        (coef_df["target"].astype(str) == target)
        & (coef_df["term"].astype(str) != "intercept")
    ].copy()

    if sub.empty:
        ax.text(0.5, 0.5, "No coefficients", ha="center", va="center")
        ax.set_axis_off()
        return

    # Keep a biologically interpretable order: adjustment terms first, then selected bridge predictors.
    order = [t for t in PREDICTOR_ORDER if t in set(sub["term"])]
    sub["term"] = pd.Categorical(sub["term"], categories=order, ordered=True)
    sub = sub.sort_values("term").reset_index(drop=True)

    y = np.arange(len(sub))
    vals = sub["coefficient"].to_numpy(dtype=float)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals]

    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([clean_term_label(t) for t in sub["term"].astype(str)])
    ax.set_xlabel("Coefficient\n(standardized predictor scale)")
    ax.set_title(f"{TARGET_DISPLAY[target]} model coefficients", pad=8)

    # Balanced x-limits
    max_abs = max(0.05, float(np.nanmax(np.abs(vals))) * 1.25)
    ax.set_xlim(-max_abs, max_abs)

    alpha = float(sub["alpha"].iloc[0])
    k_bridge = int(
        np.sum([str(t).startswith("bridge_") for t in sub["term"].astype(str)])
    )

    ax.text(
        0.03,
        0.05,
        f"α={alpha:.3g}; bridge k={k_bridge}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#dddddd",
            alpha=0.95,
        ),
    )

    add_panel_label(ax, label)


def main() -> None:
    for p in [SUMMARY_CSV, COEF_CSV]:
        require_file(p)

    summary_df = ordered_summary(pd.read_csv(SUMMARY_CSV))
    coef_df = pd.read_csv(COEF_CSV)

    required_summary = [
        "target",
        "target_col",
        "predictor_terms",
        "loocv_rmse",
        "loocv_r2",
        "alpha",
        "n_bridge_predictors",
    ]
    missing = [c for c in required_summary if c not in summary_df.columns]
    if missing:
        raise ValueError(f"Summary CSV missing columns: {missing}")

    required_coef = ["target", "term", "coefficient", "alpha"]
    missing = [c for c in required_coef if c not in coef_df.columns]
    if missing:
        raise ValueError(f"Coefficient CSV missing columns: {missing}")

    fig = plt.figure(figsize=(13.4, 11.0))
    gs = fig.add_gridspec(
        3,
        2,
        left=0.075,
        right=0.985,
        top=0.925,
        bottom=0.075,
        wspace=0.38,
        hspace=0.48,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])

    panel_a_performance(axA, summary_df)
    panel_b_selection(axB, summary_df)
    coefficient_panel(axC, coef_df, "epi_like", "C")
    coefficient_panel(axD, coef_df, "fibroblast", "D")
    coefficient_panel(axE, coef_df, "smooth_myoepi", "E")
    coefficient_panel(axF, coef_df, "ECM", "F")

#    fig.suptitle(
#        "Regularized exploratory ridge modeling of region-level bridge features",
#        fontsize=14,
#        fontweight="bold",
#        y=0.975,
#    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" ", OUT_PNG)
    print(" ", OUT_PDF)

    print("\nSelected ridge models:")
    cols = [
        "target",
        "loocv_rmse",
        "loocv_r2",
        "loocv_pearson_r",
        "alpha",
        "n_bridge_predictors",
        "bridge_predictors",
    ]
    cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
