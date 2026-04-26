from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
})


COLORS = {
    "navy": "#1f2a52",
    "panel_edge": "#d9dcef",
    "panel_bg": "#f7f8ff",
    "arrow": "#52689a",
    "crc": "#4f79b8",
    "breast": "#c783a7",
    "external": "#8c8c8c",
    "q1": "#f2c14e",
    "q2": "#4f8fcf",
    "q3": "#74a76f",
    "q4": "#d85c5c",
    "epi": "#f2c14e",
    "fib": "#72b879",
    "smooth": "#b99ac8",
    "ecm": "#4f8fcf",
    "gray": "#eeeeee",
}


def panel(ax, xy, wh, label, title):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=COLORS["panel_edge"],
        facecolor=COLORS["panel_bg"],
        zorder=0,
    )
    ax.add_patch(box)

    lab = FancyBboxPatch(
        (x + 0.010, y + h - 0.040), 0.045, 0.030,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        zorder=5,
    )
    ax.add_patch(lab)
    ax.text(
        x + 0.0325, y + h - 0.025, label,
        color="white", ha="center", va="center",
        fontsize=9, fontweight="bold", zorder=6,
    )
    ax.text(
        x + 0.065, y + h - 0.025, title,
        color=COLORS["navy"], ha="left", va="center",
        fontsize=10.5, fontweight="bold",
    )


def arrow(ax, start, end, mutation_scale=12):
    """Draw a modest workflow connector that does not obscure panel content."""
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            color=COLORS["arrow"],
            alpha=0.75,
            linewidth=1.8,
            mutation_scale=mutation_scale,
            shrinkA=4,
            shrinkB=4,
            zorder=3,
        )
    )


def mini_tissue(ax, cx, cy, w, h, color, seed=1):
    """Draw a stylized tissue section blob."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 160)
    r = 1 + 0.12 * np.sin(3 * t + seed) + 0.08 * np.cos(5 * t)
    x = cx + (w / 2) * r * np.cos(t)
    y = cy + (h / 2) * r * np.sin(t)
    ax.fill(x, y, color=color, alpha=0.28, edgecolor=COLORS["navy"], linewidth=0.8)

    # mottled texture
    for _ in range(85):
        px = rng.normal(cx, w / 4.2)
        py = rng.normal(cy, h / 4.2)
        if ((px - cx) / (w / 2)) ** 2 + ((py - cy) / (h / 2)) ** 2 < 1.0:
            ax.scatter(px, py, s=rng.uniform(3, 9), color=color, alpha=0.35, linewidth=0)


def quadrant_box(ax, x, y, w, h, title):
    """Draw median-based quadrant box with correct labels."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=COLORS["navy"], linewidth=1.0))
    ax.add_patch(Rectangle((x, y + h/2), w/2, h/2, facecolor=COLORS["q1"], alpha=0.82, edgecolor="white"))
    ax.add_patch(Rectangle((x + w/2, y + h/2), w/2, h/2, facecolor=COLORS["q2"], alpha=0.82, edgecolor="white"))
    ax.add_patch(Rectangle((x, y), w/2, h/2, facecolor=COLORS["q3"], alpha=0.82, edgecolor="white"))
    ax.add_patch(Rectangle((x + w/2, y), w/2, h/2, facecolor=COLORS["q4"], alpha=0.82, edgecolor="white"))
    ax.plot([x + w/2, x + w/2], [y, y + h], color=COLORS["navy"], lw=1.0)
    ax.plot([x, x + w], [y + h/2, y + h/2], color=COLORS["navy"], lw=1.0)

    ax.text(x + w*0.25, y + h*0.75, "Q1_UL", ha="center", va="center", fontsize=9.2, fontweight="bold", color="white")
    ax.text(x + w*0.75, y + h*0.75, "Q2_UR", ha="center", va="center", fontsize=9.2, fontweight="bold", color="white")
    ax.text(x + w*0.25, y + h*0.25, "Q3_LL", ha="center", va="center", fontsize=9.2, fontweight="bold", color="white")
    ax.text(x + w*0.75, y + h*0.25, "Q4_LR", ha="center", va="center", fontsize=9.2, fontweight="bold", color="white")
    ax.text(x + w/2, y + h + 0.015, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLORS["navy"])
    ax.text(
    x + w/2,
    y - 0.010,
    "median x/y split",
    ha="center",
    va="top",
    fontsize=7.4,
    color=COLORS["navy"],
    )


def bridge_feature_icon(ax, cx, cy, color, label):
    circ = Circle((cx, cy), 0.030, facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.95)
    ax.add_patch(circ)
    ax.text(cx, cy - 0.052, label, ha="center", va="top", fontsize=8.2, color=COLORS["navy"])


def small_matrix(ax, x, y, w, h, rows, cols, row_labels=None, col_labels=None, title=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=COLORS["navy"], linewidth=0.8))
    cell_w = w / cols
    cell_h = h / rows

    rng = np.random.default_rng(7)
    palette = [COLORS["epi"], COLORS["fib"], COLORS["smooth"], COLORS["ecm"], "#ffffff"]

    for r in range(rows):
        for c in range(cols):
            color = palette[(r + c + rng.integers(0, 3)) % len(palette)]
            ax.add_patch(
                Rectangle(
                    (x + c*cell_w, y + (rows-1-r)*cell_h),
                    cell_w, cell_h,
                    facecolor=color,
                    edgecolor="#d6d6d6",
                    linewidth=0.45,
                    alpha=0.75,
                )
            )

    if title:
        ax.text(x + w/2, y + h + 0.012, title, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLORS["navy"])


def validation_badge(ax, x, y, text, color="#ffffff", w=0.145, h=0.044, fontsize=8.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        facecolor=color,
        edgecolor=COLORS["panel_edge"],
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["navy"],
    )


# ---------------------------------------------------------------------
# Figure canvas
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(14, 9), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(
    0.5, 0.965,
    "Region-aware bridge modeling for mesoscale spatial transcriptomic representation",
    ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["navy"],
)
ax.plot([0.04, 0.96], [0.925, 0.925], color=COLORS["navy"], lw=1.0)


# Panels
panel(ax, (0.035, 0.615), (0.285, 0.275), "A", "Study design")
panel(ax, (0.340, 0.615), (0.285, 0.275), "B", "Bridge-target construction")
panel(ax, (0.645, 0.615), (0.320, 0.275), "C", "Median-quadrant partitioning")

panel(ax, (0.035, 0.320), (0.590, 0.255), "D", "Quadrant-level bridge summaries and validation")
panel(ax, (0.645, 0.320), (0.320, 0.255), "E", "Joint design matrix and exploratory\n modeling")

panel(ax, (0.035, 0.075), (0.930, 0.205), "F", "Supplementary external solid-tumor applicability tests")


# ---------------------------------------------------------------------
# Panel A: study design
# ---------------------------------------------------------------------
mini_tissue(ax, 0.125, 0.760, 0.110, 0.075, COLORS["crc"], seed=2)
mini_tissue(ax, 0.225, 0.760, 0.110, 0.075, COLORS["breast"], seed=5)

ax.text(0.125, 0.825, "CRC Visium HD", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLORS["navy"])
ax.text(0.225, 0.825, "Breast Visium HD", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLORS["navy"])
arrow(ax, (0.160, 0.760), (0.190, 0.760), mutation_scale=10)

validation_badge(ax, 0.065, 0.670, "Primary proof-of-concept", "#ffffff", w=0.110, h=0.038, fontsize=7.4)
validation_badge(ax, 0.065, 0.620, "1 section each", "#ffffff", w=0.110, h=0.038, fontsize=7.4)
validation_badge(ax, 0.188, 0.620, "within-section validation", "#ffffff", w=0.110, h=0.038, fontsize=7.4)


# ---------------------------------------------------------------------
# Panel B: bridge target construction
# ---------------------------------------------------------------------
ax.text(0.482, 0.825, "Transparent 21D bridge-target layer", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=COLORS["navy"])

validation_badge(ax, 0.365, 0.758, "coarse cell-state\nindicators", "#ffffff", w=0.070, h=0.038, fontsize=7.4)
validation_badge(ax, 0.455, 0.758, "curated gene-\nprogram scores", "#ffffff", w=0.070, h=0.038, fontsize=7.4)
validation_badge(ax, 0.545, 0.758, "QC summaries", "#ffffff", w=0.070, h=0.038, fontsize=7.4)

ax.text(0.482, 0.730, "Core-four features used for region-aware aggregation", ha="center",
        va="center", fontsize=8.8, color=COLORS["navy"])

bridge_feature_icon(ax, 0.390, 0.687, COLORS["epi"], "Epithelial-\nlike")
bridge_feature_icon(ax, 0.455, 0.687, COLORS["fib"], "Fibroblast")
bridge_feature_icon(ax, 0.525, 0.687, COLORS["smooth"], "Smooth/\nmyoepithelial")
bridge_feature_icon(ax, 0.595, 0.687, COLORS["ecm"], "ECM")


# ---------------------------------------------------------------------
# Panel C: median quadrants
# ---------------------------------------------------------------------
quadrant_box(ax, 0.685, 0.700, 0.115, 0.105, "CRC")
quadrant_box(ax, 0.835, 0.700, 0.115, 0.105, "Breast")

ax.text(0.817, 0.650, "Same labels in both sections", ha="center", va="center",
        fontsize=9, fontweight="bold", color=COLORS["navy"])
ax.text(0.817, 0.625, "Q1_UL, Q2_UR, Q3_LL, Q4_LR", ha="center", va="center",
        fontsize=8.5, color=COLORS["navy"])


# Top workflow arrows
arrow(ax, (0.320, 0.752), (0.340, 0.752), mutation_scale=11)
arrow(ax, (0.625, 0.752), (0.645, 0.752), mutation_scale=11)


# ---------------------------------------------------------------------
# Panel D: regional summaries and validation
# ---------------------------------------------------------------------
quadrant_box(ax, 0.070, 0.405, 0.115, 0.100, "CRC regions")
quadrant_box(ax, 0.220, 0.405, 0.115, 0.100, "Breast regions")

arrow(ax, (0.345, 0.455), (0.390, 0.455), mutation_scale=10)

small_matrix(ax, 0.405, 0.405, 0.150, 0.100, rows=4, cols=4, title="Region × bridge matrix")

ax.text(0.480, 0.395, "4 features × 4 regions", ha="center", va="top", fontsize=8, color=COLORS["navy"])

validation_badge(ax, 0.083, 0.332, "whole-section\ncomparison", "#ffffff", w=0.085, h=0.036, fontsize=7.4)
validation_badge(ax, 0.234, 0.332, "shuffle-null\nvalidation", "#ffffff", w=0.085, h=0.036, fontsize=7.4)
validation_badge(ax, 0.436, 0.332, "shifted/rotated\npartitions", "#ffffff", w=0.085, h=0.036, fontsize=7.4)

ax.text(0.5290, 0.332, "Output: region-level\nbridge summary tables",
        ha="left", va="center", fontsize=8.0, color=COLORS["navy"])


# ---------------------------------------------------------------------
# Panel E: design matrix and modeling
# ---------------------------------------------------------------------
small_matrix(ax, 0.675, 0.408, 0.145, 0.090, rows=8, cols=7, title="Eight-row design matrix")

ax.text(0.747, 0.400, "8-row CRC--Breast\nmedian-quadrant matrix",
        ha="center", va="top", fontsize=7.6, color=COLORS["navy"])

arrow(ax, (0.825, 0.455), (0.855, 0.455), mutation_scale=10)

# Ridge icon
ax.add_patch(Rectangle((0.865, 0.430), 0.045, 0.060, facecolor="white", edgecolor=COLORS["panel_edge"]))
ax.plot([0.870, 0.900], [0.437, 0.477], color=COLORS["navy"], lw=1.0)
ax.scatter([0.872, 0.883, 0.895], [0.445, 0.462, 0.475], s=18, color=[COLORS["epi"], COLORS["fib"], COLORS["q4"]])
ax.text(0.888, 0.410, "Ridge\nexploratory", ha="center", va="top", fontsize=7.6, color=COLORS["navy"])

# Bayesian icon
ax.add_patch(Rectangle((0.920, 0.430), 0.035, 0.060, facecolor="white", edgecolor=COLORS["panel_edge"]))
ax.plot([0.926, 0.948, 0.936], [0.445, 0.445, 0.477], color=COLORS["navy"], lw=1.0)
ax.scatter([0.926, 0.948, 0.936], [0.445, 0.445, 0.477], s=[28, 28, 50], color=[COLORS["q2"], COLORS["q4"], COLORS["ecm"]])
ax.text(0.938, 0.410, "Bayesian\nsensitivity", ha="center", va="top", fontsize=7.6, color=COLORS["navy"])

validation_badge(ax, 0.697, 0.324, "8 regions\n4 CRC + 4 Breast", "#ffffff", w=0.100, h=0.038, fontsize=7.4)
validation_badge(ax, 0.859, 0.324, "exploratory\nnot confirmatory", "#ffffff", w=0.100, h=0.038, fontsize=7.4)


# ---------------------------------------------------------------------
# Panel F: external solid tumor applicability tests
# ---------------------------------------------------------------------
ax.text(0.130, 0.235, "Supplementary only", ha="left", va="center",
        fontsize=10, fontweight="bold", color=COLORS["navy"])

for i, (label, color, seed) in enumerate([
    ("Lung cancer", "#9bbcd8", 10),
    ("Prostate cancer", "#b9a1d6", 11),
    ("Ovarian cancer", "#e0a4a4", 12),
]):
    x0 = 0.170 + i * 0.130
    mini_tissue(ax, x0, 0.185, 0.075, 0.050, color, seed=seed)
    ax.text(x0, 0.135, label, ha="center", va="top", fontsize=8.5, color=COLORS["navy"])

arrow(ax, (0.490, 0.185), (0.585, 0.185), mutation_scale=10)

validation_badge(ax, 0.610, 0.198, "marker-program\nbridge scoring", "#ffffff", w=0.110, h=0.045, fontsize=7.4)
validation_badge(ax, 0.740, 0.198, "median-quadrant\naggregation", "#ffffff", w=0.110, h=0.045, fontsize=7.4)
validation_badge(ax, 0.610, 0.128, "shuffle-null\nvalidation", "#ffffff", w=0.110, h=0.045, fontsize=7.4)
validation_badge(ax, 0.740, 0.128, "partition\nsensitivity", "#ffffff", w=0.110, h=0.045, fontsize=7.4)

ax.text(0.910, 0.190, "Not included in\nCRC--Breast ridge/\nBayesian models",
        ha="center", va="center", fontsize=8.2, color=COLORS["navy"],
        bbox=dict(boxstyle="round,pad=0.39", facecolor="#ffffff", edgecolor=COLORS["panel_edge"]))


# Bottom footnote
#ax.text(
#    0.5, 0.035,
#    "All active revision outputs use clean region-aware bridge-modeling filenames; legacy OU/branching names are not used in the revised workflow.",
#    ha="center", va="center", fontsize=8, color="#555555",
#)


# Save
png_out = OUTDIR / "Figure1_revised_region_aware_bridge_workflow.png"
pdf_out = OUTDIR / "Figure1_revised_region_aware_bridge_workflow.pdf"

fig.savefig(png_out, dpi=500, bbox_inches="tight")
fig.savefig(pdf_out, bbox_inches="tight")
plt.close(fig)

print(f"[OK] wrote {png_out}")
print(f"[OK] wrote {pdf_out}")
