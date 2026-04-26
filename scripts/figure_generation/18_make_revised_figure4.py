from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POSTERIOR_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "08_bayesian_models"
    / "bayesian_posterior_summary.csv"
)

DIAG_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "08_bayesian_models"
    / "bayesian_diagnostics.csv"
)

CONFIG_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "08_bayesian_models"
    / "bayesian_model_config.csv"
)

OUTDIR = PROJECT_ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Figure4_revised_bayesian_sensitivity_modeling.png"
OUT_PDF = OUTDIR / "Figure4_revised_bayesian_sensitivity_modeling.pdf"


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


TARGET_DISPLAY = {
    "epi_like": "Epithelial-like",
    "fibroblast": "Fibroblast",
    "smooth_myoepi": "Smooth/Myoepithelial",
    "ECM": "ECM",
}

TARGET_SHORT = {
    "epi_like": "Epi-like",
    "fibroblast": "Fibroblast",
    "smooth_myoepi": "Smooth/Myo",
    "ECM": "ECM",
}

TERM_DISPLAY = {
    "bridge_epi_like_mean_joint_z": "Epithelial-like",
    "bridge_fibroblast_mean_joint_z": "Fibroblast",
    "bridge_smooth_myoepi_mean_joint_z": "Smooth/Myoepithelial",
    "bridge_ECM_mean_joint_z": "ECM",
    "dataset_indicator_breast": "Breast indicator",
    "region_indicator_Q2_UR": "Q2_UR",
    "region_indicator_Q3_LL": "Q3_LL",
    "region_indicator_Q4_LR": "Q4_LR",
}

TARGET_ORDER = ["epi_like", "fibroblast", "smooth_myoepi", "ECM"]

BRIDGE_TERMS = [
    "bridge_epi_like_mean_joint_z",
    "bridge_fibroblast_mean_joint_z",
    "bridge_smooth_myoepi_mean_joint_z",
    "bridge_ECM_mean_joint_z",
]

COLORS = {
    "positive": "#4f79b8",
    "negative": "#d95f02",
    "neutral": "#777777",
    "node_epi": "#f2c14e",
    "node_fib": "#72b879",
    "node_smooth": "#b99ac8",
    "node_ecm": "#4f8fcf",
    "navy": "#1f2a52",
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


def hdi_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("hdi_3%", "hdi_97%"),
        ("hdi_3.0%", "hdi_97.0%"),
        ("hdi_3", "hdi_97"),
        ("hdi_lower", "hdi_upper"),
    ]
    for lo, hi in candidates:
        if lo in df.columns and hi in df.columns:
            return lo, hi
    raise ValueError(f"Could not identify HDI columns in: {list(df.columns)}")


def clean_posterior(post: pd.DataFrame) -> pd.DataFrame:
    lo_col, hi_col = hdi_columns(post)
    out = post.copy()
    out["hdi_low"] = pd.to_numeric(out[lo_col], errors="coerce")
    out["hdi_high"] = pd.to_numeric(out[hi_col], errors="coerce")
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    out["sd"] = pd.to_numeric(out["sd"], errors="coerce")
    out["term"] = out["term"].astype(str)
    out["target"] = out["target"].astype(str)
    return out


def bridge_only(post: pd.DataFrame) -> pd.DataFrame:
    out = post[post["term"].isin(BRIDGE_TERMS)].copy()
    out["target_display"] = out["target"].map(TARGET_DISPLAY)
    out["term_display"] = out["term"].map(TERM_DISPLAY)
    out["label"] = out["target_display"] + " \u2190 " + out["term_display"]

    target_rank = {t: i for i, t in enumerate(TARGET_ORDER)}
    term_rank = {t: i for i, t in enumerate(BRIDGE_TERMS)}

    out["target_rank"] = out["target"].map(target_rank)
    out["term_rank"] = out["term"].map(term_rank)
    out = out.sort_values(["target_rank", "term_rank"]).reset_index(drop=True)
    return out


def select_row(post: pd.DataFrame, target: str, term: str) -> pd.Series:
    sub = post[(post["target"] == target) & (post["term"] == term)].copy()
    if sub.empty:
        raise ValueError(f"Missing posterior row for target={target}, term={term}")
    return sub.iloc[0]


def plot_interval(ax, row, label: str, title: str, color: str | None = None):
    mean = float(row["mean"])
    lo = float(row["hdi_low"])
    hi = float(row["hdi_high"])

    if color is None:
        color = COLORS["positive"] if mean >= 0 else COLORS["negative"]

    ax.axvline(0, color="black", linewidth=1.0)
    ax.errorbar(
        [mean],
        [0],
        xerr=[[mean - lo], [hi - mean]],
        fmt="o",
        capsize=5,
        linewidth=2.2,
        markersize=6,
        color=color,
        ecolor=color,
    )
    ax.set_yticks([0])
    ax.set_yticklabels([label])
    ax.set_xlabel("Posterior coefficient with 94% HDI")
    ax.set_title(title, pad=8)

    span = max(abs(lo), abs(hi), 0.5) * 1.25
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.7, 0.7)

    ax.text(
        0.03,
        0.08,
        f"mean={mean:.3f}\n94% HDI [{lo:.3f}, {hi:.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#dddddd"),
    )


def panel_a_forest(ax, bridge_df: pd.DataFrame):
    y = np.arange(len(bridge_df))[::-1]
    means = bridge_df["mean"].to_numpy(dtype=float)
    lows = bridge_df["hdi_low"].to_numpy(dtype=float)
    highs = bridge_df["hdi_high"].to_numpy(dtype=float)

    colors = [COLORS["positive"] if m >= 0 else COLORS["negative"] for m in means]

    ax.axvline(0, color="black", linewidth=1.0)
    for yi, mean, lo, hi, color in zip(y, means, lows, highs, colors):
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            capsize=3,
            linewidth=1.5,
            markersize=5,
            color=color,
            ecolor=color,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(bridge_df["label"].tolist())
    ax.set_xlabel("Posterior coefficient with 94% HDI")
    ax.set_title("Bridge-feature coefficient summaries", pad=8)
    add_panel_label(ax, "A")


def panel_d_diagnostics(ax, diag_df: pd.DataFrame):
    diag = diag_df.copy()
    diag["target"] = diag["target"].astype(str)

    agg = (
        diag.groupby("target")
        .agg(max_r_hat=("r_hat", "max"), min_ess_bulk=("ess_bulk", "min"))
        .reindex(TARGET_ORDER)
        .reset_index()
    )

    x = np.arange(len(agg))
    ax2 = ax.twinx()

    bars = ax.bar(
        x - 0.15,
        agg["max_r_hat"].to_numpy(dtype=float),
        width=0.30,
        color="#bdbdbd",
        label="max R-hat",
    )
    line = ax2.plot(
        x + 0.15,
        agg["min_ess_bulk"].to_numpy(dtype=float),
        marker="o",
        color=COLORS["navy"],
        linewidth=1.8,
        label="min bulk ESS",
    )

    ax.axhline(1.01, color="#d95f02", linestyle="--", linewidth=1.0)
    ax.text(
        0.02,
        0.94,
        "R-hat guideline=1.01",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#d95f02",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_SHORT[t] for t in agg["target"]])
    ax.set_ylabel("Max R-hat")
    ax2.set_ylabel("Min bulk ESS")
    ax.set_title("Bayesian sampling diagnostics", pad=8)

    # Combine legends.
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper center", fontsize=8)

    # Keep Rhat axis near relevant range.
    max_rhat = float(np.nanmax(agg["max_r_hat"]))
    ax.set_ylim(0.98, max(1.04, max_rhat + 0.015))

    add_panel_label(ax, "D")


def ellipse_boundary_point(center, target, a, b):
    cx, cy = center
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    scale = 1.0 / np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    return cx + scale * dx, cy + scale * dy


def draw_arrow(ax, start, end, color, rad=0.0, linestyle="-", linewidth=2.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arr)


def panel_e_concept(ax):
    ax.set_axis_off()
    
    # Expanded limits prevent large ovals and annotations from being clipped.
    ax.set_xlim(-0.30, 1.15)
    ax.set_ylim(-0.08, 1.10)
    ax.set_aspect("equal")
    
    ax.set_title("Conservative bridge-association summary", pad=8)
    
    # Manual panel label, shifted farther left than the default helper.
    ax.text(
        -0.17,
        1.03,
        "E",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="right",
        va="top",
        clip_on=False,
    )

    pos = {
        "Epithelial-like": (0.02, 0.90),
        "Fibroblast": (0.88, 0.90),
        "Smooth/Myo": (0.07, 0.35),
        "ECM": (0.88, 0.35),
    }
    node_colors = {
        "Epithelial-like": COLORS["node_epi"],
        "Fibroblast": COLORS["node_fib"],
        "Smooth/Myo": COLORS["node_smooth"],
        "ECM": COLORS["node_ecm"],
    }

    oval_w = 0.47
    oval_h = 0.28
    a = oval_w / 2
    b = oval_h / 2

    for name, (x, y) in pos.items():
        node = Ellipse(
            (x, y),
            width=oval_w,
            height=oval_h,
            facecolor=node_colors[name],
            edgecolor="#444444",
            linewidth=1.2,
            alpha=0.55,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(
            x, y, name,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            zorder=4,
        )

    # Clean supported candidate: Fibroblast -> ECM
    start = ellipse_boundary_point(pos["Fibroblast"], pos["ECM"], a, b)
    end = ellipse_boundary_point(pos["ECM"], pos["Fibroblast"], a, b)
    draw_arrow(
        ax,
        start,
        end,
        color=COLORS["positive"],
        rad=-0.18,
        linewidth=2.5,
    )
    ax.text(
        0.97, 0.645, "+",
        color=COLORS["positive"],
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        zorder=5,
    )

    # Exploratory epithelial--fibroblast sensitivity, shown as one dashed double-headed line
    start = ellipse_boundary_point(pos["Epithelial-like"], pos["Fibroblast"], a, b)
    end = ellipse_boundary_point(pos["Fibroblast"], pos["Epithelial-like"], a, b)

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<|-|>",
            mutation_scale=12,
            linewidth=1.8,
            color=COLORS["negative"],
            linestyle="--",
            connectionstyle="arc3,rad=0.00",
            zorder=2,
        )
    )
    ax.text(
        0.45, 0.96, "−",
        color=COLORS["negative"],
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        zorder=5,
    )

    # Make Smooth/Myo intentionally unconnected in this conservative summary
    ax.text(
        0.10, 0.13,
        "no bridge predictor\nselected",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#555555",
    )

    ax.text(
        0.50,
        -0.05,
        "Solid arrow: clean ECM-model association\n"
        "Dashed double arrow: exploratory sensitivity association",
        ha="center",
        va="center",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#dddddd",
            alpha=0.95,
        ),
    )


def main() -> None:
    for p in [POSTERIOR_CSV, DIAG_CSV, CONFIG_CSV]:
        require_file(p)

    post = clean_posterior(pd.read_csv(POSTERIOR_CSV))
    diag = pd.read_csv(DIAG_CSV)
    config = pd.read_csv(CONFIG_CSV)

    bridge_df = bridge_only(post)

    required = ["target", "term", "mean", "hdi_low", "hdi_high", "r_hat", "ess_bulk"]
    missing = [c for c in required if c not in post.columns]
    if missing:
        raise ValueError(f"Posterior CSV missing columns: {missing}")

    fig = plt.figure(figsize=(13.2, 11.0))
    gs = fig.add_gridspec(
        3,
        2,
        left=0.105,
        right=0.985,
        top=0.925,
        bottom=0.075,
        wspace=0.42,
        hspace=0.50,
        height_ratios=[1.10, 1.0, 1.05],
    )

    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 0])
    axE = fig.add_subplot(gs[2, 1])

    panel_a_forest(axA, bridge_df)

    # Main clean candidate.
    row_ecm_fib = select_row(post, "ECM", "bridge_fibroblast_mean_joint_z")
    plot_interval(
        axB,
        row_ecm_fib,
        "ECM \u2190 Fibroblast",
        "Clean candidate association: fibroblast to ECM",
        color=COLORS["positive"],
    )
    add_panel_label(axB, "B")

    # Exploratory epithelial/fibroblast sensitivity associations.
    row_fib_epi = select_row(post, "fibroblast", "bridge_epi_like_mean_joint_z")
    row_epi_fib = select_row(post, "epi_like", "bridge_fibroblast_mean_joint_z")

    y = np.array([1, 0])
    means = np.array([row_fib_epi["mean"], row_epi_fib["mean"]], dtype=float)
    lows = np.array([row_fib_epi["hdi_low"], row_epi_fib["hdi_low"]], dtype=float)
    highs = np.array([row_fib_epi["hdi_high"], row_epi_fib["hdi_high"]], dtype=float)
    labels = ["Fibroblast \u2190 Epithelial-like", "Epithelial-like \u2190 Fibroblast"]

    axC.axvline(0, color="black", linewidth=1.0)
    for yi, mean, lo, hi in zip(y, means, lows, highs):
        axC.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            capsize=5,
            linewidth=2.0,
            markersize=6,
            color=COLORS["negative"],
            ecolor=COLORS["negative"],
        )

    axC.set_yticks(y)
    axC.set_yticklabels(labels)
    axC.set_xlabel("Posterior coefficient with 94% HDI")
    axC.set_title("Exploratory epithelial--fibroblast sensitivity", pad=8)
    span = max(np.max(np.abs(lows)), np.max(np.abs(highs)), 0.5) * 1.20
    axC.set_xlim(-span, span)

    axC.text(
        0.54,
        0.08,
        "Interpreted cautiously:\nparameter-rich models retained\nsome divergent transitions.",
        transform=axC.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#dddddd"),
    )
    add_panel_label(axC, "C")

    panel_d_diagnostics(axD, diag)
    panel_e_concept(axE)

#    fig.suptitle(
#        "Bayesian sensitivity modeling of region-level bridge-feature associations",
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

    print("\nBridge coefficient posterior summaries:")
    cols = ["target", "term", "mean", "sd", "hdi_low", "hdi_high", "r_hat", "ess_bulk"]
    print(bridge_df[cols].to_string(index=False))

    print("\nBayesian model config:")
    config_cols = [
        "target",
        "draws",
        "tune",
        "chains",
        "target_accept",
        "prior_beta",
        "prior_sigma",
    ]
    config_cols = [c for c in config_cols if c in config.columns]
    print(config[config_cols].to_string(index=False))


if __name__ == "__main__":
    main()
