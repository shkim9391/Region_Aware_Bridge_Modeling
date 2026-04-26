from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "04_region_summaries"
    / "combined_region_bridge_summary_crc_breast_core4_regions4.csv"
)

HET_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "05_validation"
    / "within_section_heterogeneity_metrics_by_scheme.csv"
)

SHUFFLE_RAW_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "05_validation"
    / "within_section_shuffle_null_raw_long.csv"
)

SHUFFLE_SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "05_validation"
    / "within_section_shuffle_null_summary.csv"
)

OUTDIR = PROJECT_ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Figure2_revised_within_section_validation.png"
OUT_PDF = OUTDIR / "Figure2_revised_within_section_validation.pdf"


# --------------------------------------------------
# Style
# --------------------------------------------------
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


DATASET_ORDER = ["CRC", "Breast"]
DATASET_COLORS = {
    "CRC": "#1f77b4",
    "Breast": "#ff7f0e",
}

REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]
REGION_DISPLAY = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]

FEATURES = [
    ("bridge_epi_like_mean", "Epithelial-like"),
    ("bridge_fibroblast_mean", "Fibroblast"),
    ("bridge_smooth_myoepi_mean", "Smooth/\nMyoepithelial"),
    ("bridge_ECM_mean", "ECM"),
]

PANEL_C_METRICS = [
    ("region_range", "Regional\nrange"),
    ("mean_abs_pairwise_diff", "Mean abs.\ninter-region\ndiff."),
    ("mean_abs_deviation_from_global", "Mean abs. deviation\nfrom whole-section\nmean"),
]

SCHEME_ORDER = ["original", "shifted", "rotated"]
SCHEME_DISPLAY = ["median", "shifted", "rotated"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
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


def format_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.4f}"


def get_original_overall_het(het_df: pd.DataFrame, dataset: str, metric: str) -> float:
    row = het_df[
        (het_df["dataset"].astype(str) == dataset)
        & (het_df["scheme"].astype(str) == "original")
        & (het_df["feature"].astype(str) == "__overall__")
    ]
    if row.empty:
        raise ValueError(f"Missing heterogeneity row for {dataset}, original, __overall__, {metric}")
    return float(row[metric].iloc[0])


def get_scheme_overall_het(het_df: pd.DataFrame, dataset: str, scheme: str, metric: str) -> float:
    row = het_df[
        (het_df["dataset"].astype(str) == dataset)
        & (het_df["scheme"].astype(str) == scheme)
        & (het_df["feature"].astype(str) == "__overall__")
    ]
    if row.empty:
        raise ValueError(f"Missing heterogeneity row for {dataset}, {scheme}, __overall__, {metric}")
    return float(row[metric].iloc[0])


def make_heatmap_matrix(region_df: pd.DataFrame, dataset: str) -> np.ndarray:
    sub = region_df[region_df["dataset"].astype(str) == dataset].copy()
    if sub.empty:
        raise ValueError(f"No region rows found for dataset={dataset}")

    sub["region_id"] = pd.Categorical(sub["region_id"], categories=REGION_ORDER, ordered=True)
    sub = sub.sort_values("region_id")

    rows = []
    for region in REGION_ORDER:
        r = sub[sub["region_id"] == region]
        if r.empty:
            raise ValueError(f"Missing {region} for dataset={dataset}")
        rows.append([float(r[col].iloc[0]) for col, _ in FEATURES])

    return np.asarray(rows, dtype=float)


def get_shuffle_data(raw_df: pd.DataFrame, summary_df: pd.DataFrame, dataset: str):
    metric = "mean_abs_pairwise_diff"

    raw = raw_df[
        (raw_df["dataset"].astype(str) == dataset)
        & (raw_df["feature"].astype(str) == "__overall__")
        & (raw_df["metric_name"].astype(str) == metric)
    ].copy()

    summ = summary_df[
        (summary_df["dataset"].astype(str) == dataset)
        & (summary_df["feature"].astype(str) == "__overall__")
        & (summary_df["metric_name"].astype(str) == metric)
    ].copy()

    if raw.empty:
        raise ValueError(f"No shuffle raw values for {dataset}")
    if summ.empty:
        raise ValueError(f"No shuffle summary for {dataset}")

    vals = raw["value"].dropna().to_numpy(dtype=float)
    obs = float(summ["observed_value"].iloc[0])
    null_mean = float(summ["null_mean"].iloc[0])
    pval = float(summ["empirical_p_upper"].iloc[0])
    return vals, obs, null_mean, pval


# --------------------------------------------------
# Load data
# --------------------------------------------------
for p in [REGION_CSV, HET_CSV, SHUFFLE_RAW_CSV, SHUFFLE_SUMMARY_CSV]:
    require_file(p)

region_df = pd.read_csv(REGION_CSV)
het_df = pd.read_csv(HET_CSV)
shuffle_raw_df = pd.read_csv(SHUFFLE_RAW_CSV)
shuffle_summary_df = pd.read_csv(SHUFFLE_SUMMARY_CSV)


# --------------------------------------------------
# Prepare data
# --------------------------------------------------
mat_crc = make_heatmap_matrix(region_df, "CRC")
mat_breast = make_heatmap_matrix(region_df, "Breast")

shared_vmin = float(np.nanmin([np.nanmin(mat_crc), np.nanmin(mat_breast)]))
shared_vmax = float(np.nanmax([np.nanmax(mat_crc), np.nanmax(mat_breast)]))

panel_c_crc = [get_original_overall_het(het_df, "CRC", metric) for metric, _ in PANEL_C_METRICS]
panel_c_breast = [get_original_overall_het(het_df, "Breast", metric) for metric, _ in PANEL_C_METRICS]

range_crc = [get_scheme_overall_het(het_df, "CRC", s, "region_range") for s in SCHEME_ORDER]
range_breast = [get_scheme_overall_het(het_df, "Breast", s, "region_range") for s in SCHEME_ORDER]

pair_crc = [get_scheme_overall_het(het_df, "CRC", s, "mean_abs_pairwise_diff") for s in SCHEME_ORDER]
pair_breast = [get_scheme_overall_het(het_df, "Breast", s, "mean_abs_pairwise_diff") for s in SCHEME_ORDER]

crc_null, crc_obs, crc_null_mean, crc_p = get_shuffle_data(shuffle_raw_df, shuffle_summary_df, "CRC")
breast_null, breast_obs, breast_null_mean, breast_p = get_shuffle_data(shuffle_raw_df, shuffle_summary_df, "Breast")


# --------------------------------------------------
# Build figure
# --------------------------------------------------
fig = plt.figure(figsize=(13.2, 10.2))

gs = fig.add_gridspec(
    2,
    3,
    left=0.055,
    right=0.985,
    top=0.92,
    bottom=0.080,
    wspace=0.34,
    hspace=0.34,
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1])
axF = fig.add_subplot(gs[1, 2])


# --------------------------------------------------
# Panels A and B: regional heatmaps
# --------------------------------------------------
imA = axA.imshow(mat_crc, aspect="auto", cmap="viridis", vmin=shared_vmin, vmax=shared_vmax)
axA.set_title("CRC median-quadrant regional means", pad=8)
axA.set_xticks(np.arange(len(FEATURES)))
axA.set_xticklabels([lab for _, lab in FEATURES], rotation=38, ha="right", rotation_mode="anchor")
axA.set_yticks(np.arange(len(REGION_DISPLAY)))
axA.set_yticklabels(REGION_DISPLAY)
axA.set_ylabel("Region")
add_panel_label(axA, "A")

imB = axB.imshow(mat_breast, aspect="auto", cmap="viridis", vmin=shared_vmin, vmax=shared_vmax)
axB.set_title("Breast median-quadrant regional means", pad=8)
axB.set_xticks(np.arange(len(FEATURES)))
axB.set_xticklabels([lab for _, lab in FEATURES], rotation=38, ha="right", rotation_mode="anchor")
axB.set_yticks(np.arange(len(REGION_DISPLAY)))
axB.set_yticklabels(REGION_DISPLAY)
axB.set_ylabel("Region")
add_panel_label(axB, "B")

cbar = fig.colorbar(imB, ax=[axA, axB], fraction=0.026, pad=0.020)
cbar.set_label("Mean bridge value")


# --------------------------------------------------
# Panel C: heterogeneity bar summary
# --------------------------------------------------
x = np.arange(len(PANEL_C_METRICS))
width = 0.34

bars_crc = axC.bar(
    x - width / 2,
    panel_c_crc,
    width,
    color=DATASET_COLORS["CRC"],
    label="CRC",
)

bars_breast = axC.bar(
    x + width / 2,
    panel_c_breast,
    width,
    color=DATASET_COLORS["Breast"],
    label="Breast",
)

axC.set_xticks(x)
axC.set_xticklabels([lab for _, lab in PANEL_C_METRICS], fontsize=8)
axC.tick_params(axis="x", pad=6)
for tick in axC.get_xticklabels():
    tick.set_ha("center")
    tick.set_linespacing(0.95)

axC.set_ylabel("Value")
axC.set_title("Median-quadrant heterogeneity summary", pad=8)
axC.legend(frameon=False, loc="upper right")

ymax = max(panel_c_crc + panel_c_breast) * 1.18
axC.set_ylim(0, ymax)

for bars in [bars_crc, bars_breast]:
    for b in bars:
        h = b.get_height()
        axC.text(
            b.get_x() + b.get_width() / 2,
            h + ymax * 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
        )

add_panel_label(axC, "C")


# --------------------------------------------------
# Panel D: shuffle null
# --------------------------------------------------
axD.set_title("Region-label shuffle null", pad=8)

# Plot the null distributions near zero and observed vertical lines.
bins = 40
axD.hist(
    crc_null,
    bins=bins,
    alpha=0.65,
    color=DATASET_COLORS["CRC"],
    label="CRC shuffle null",
)
axD.hist(
    breast_null,
    bins=bins,
    alpha=0.55,
    color=DATASET_COLORS["Breast"],
    label="Breast shuffle null",
)

axD.axvline(crc_obs, color=DATASET_COLORS["CRC"], linestyle="--", linewidth=2.0)
axD.axvline(breast_obs, color=DATASET_COLORS["Breast"], linestyle="--", linewidth=2.0)

xmax = max(crc_obs, breast_obs) * 1.08
axD.set_xlim(-0.005, xmax)

axD.set_xlabel("Overall mean abs. inter-region difference")
axD.set_ylabel("Permutation count")
axD.legend(
    frameon=False,
    loc="upper left",
    fontsize=8,
    bbox_to_anchor=(0.02, 0.98),
)

axD.text(
    0.50,
    0.78,
    f"CRC observed={crc_obs:.4f}\n"
    f"null mean={crc_null_mean:.4f}, P={format_p(crc_p)}\n\n"
    f"Breast observed={breast_obs:.4f}\n"
    f"null mean={breast_null_mean:.4f}, P={format_p(breast_p)}",
    transform=axD.transAxes,
    ha="center",
    va="top",
    fontsize=8.2,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#dddddd",
        alpha=0.95,
    ),
)

add_panel_label(axD, "D")


# --------------------------------------------------
# Panel E: partition sensitivity, regional range
# --------------------------------------------------
axE.plot(
    SCHEME_DISPLAY,
    range_crc,
    marker="o",
    linewidth=1.8,
    markersize=5,
    color=DATASET_COLORS["CRC"],
    label="CRC",
)
axE.plot(
    SCHEME_DISPLAY,
    range_breast,
    marker="o",
    linewidth=1.8,
    markersize=5,
    color=DATASET_COLORS["Breast"],
    label="Breast",
)

axE.set_title("Partition sensitivity:\nregional range", pad=8)
axE.set_ylabel("Value")
axE.set_xlabel("Partition scheme")
axE.legend(frameon=False, loc="best")
add_panel_label(axE, "E")


# --------------------------------------------------
# Panel F: partition sensitivity, mean abs pairwise
# --------------------------------------------------
axF.plot(
    SCHEME_DISPLAY,
    pair_crc,
    marker="o",
    linewidth=1.8,
    markersize=5,
    color=DATASET_COLORS["CRC"],
    label="CRC",
)
axF.plot(
    SCHEME_DISPLAY,
    pair_breast,
    marker="o",
    linewidth=1.8,
    markersize=5,
    color=DATASET_COLORS["Breast"],
    label="Breast",
)

axF.set_title("Partition sensitivity:\nmean abs. inter-region difference", pad=8)
axF.set_ylabel("Value")
axF.set_xlabel("Partition scheme")
axF.legend(frameon=False, loc="best")
add_panel_label(axF, "F")


# --------------------------------------------------
# Figure-level title and save
# --------------------------------------------------
#fig.suptitle(
#    "Within-section validation of median-quadrant region-aware bridge summaries",
#    fontsize=14,
#    fontweight="bold",
#    y=0.975,
#)

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" ", OUT_PNG)
print(" ", OUT_PDF)

print("\nKey values:")
print(f"CRC overall mean abs. pairwise diff: observed={crc_obs:.6f}, null_mean={crc_null_mean:.6f}, p={crc_p:.6g}")
print(f"Breast overall mean abs. pairwise diff: observed={breast_obs:.6f}, null_mean={breast_null_mean:.6f}, p={breast_p:.6g}")
