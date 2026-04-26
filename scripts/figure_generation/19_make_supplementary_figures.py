from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTDIR = PROJECT_ROOT / "outputs" / "figures" / "supplementary"
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIMARY_VALIDATION = PROJECT_ROOT / "outputs" / "05_validation"
PRIMARY_REGIONS = PROJECT_ROOT / "outputs" / "04_region_summaries"

EXTERNAL_ROOT = PROJECT_ROOT / "outputs" / "external_solid_tumors"
EXTERNAL_VALIDATION = EXTERNAL_ROOT / "03_validation"
EXTERNAL_REGIONS = EXTERNAL_ROOT / "02_region_summaries"

REGION_ORDER = ["Q1_UL", "Q2_UR", "Q3_LL", "Q4_LR"]

PRIMARY_FEATURES = [
    ("epi_like", "Epithelial-like"),
    ("fibroblast", "Fibroblast"),
    ("smooth_myoepi", "Smooth/\nMyoepithelial"),
    ("ECM", "ECM"),
]

PRIMARY_REGION_MEAN_COLS = [
    ("bridge_epi_like_mean", "Epithelial-like"),
    ("bridge_fibroblast_mean", "Fibroblast"),
    ("bridge_smooth_myoepi_mean", "Smooth/\nMyoepithelial"),
    ("bridge_ECM_mean", "ECM"),
]

EXTERNAL_PROGRAMS = [
    ("epithelial_like", "Epithelial-like"),
    ("fibroblast_stromal", "Fibroblast/\nstromal"),
    ("smooth_contractile", "Smooth/\ncontractile"),
    ("ECM", "ECM"),
]

DATASET_COLORS = {
    "CRC": "#1f77b4",
    "Breast": "#ff7f0e",
    "Lung cancer": "#1f77b4",
    "Prostate cancer": "#9467bd",
    "Ovarian cancer": "#d62728",
}

REGION_COLORS = {
    "Q1_UL": "#f2c14e",
    "Q2_UR": "#4f8fcf",
    "Q3_LL": "#74a76f",
    "Q4_LR": "#d85c5c",
}

SCHEME_ORDER = ["original", "shifted", "rotated"]
SCHEME_LABELS = ["median", "shifted", "rotated"]


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 14,
})


def add_panel_label(ax, label, x=-0.11, y=1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="right",
        va="top",
    )


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def format_p(p):
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.4f}"


# ==================================================
# Supplementary Figure S1
# ==================================================

def make_supplementary_figure_s1():
    points_csv = PRIMARY_VALIDATION / "within_section_original_partition_sampled_points.csv"
    region_csv = PRIMARY_REGIONS / "combined_region_bridge_summary_crc_breast_core4_regions4.csv"

    require_file(points_csv)
    require_file(region_csv)

    points = pd.read_csv(points_csv)
    regions = pd.read_csv(region_csv)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    axA, axB, axC, axD = axes.flatten()

    # A-B: sampled points colored by region
    for ax, dataset, label in [(axA, "CRC", "A"), (axB, "Breast", "B")]:
        sub = points[points["dataset"].astype(str) == dataset].copy()
        for region in REGION_ORDER:
            g = sub[sub["region_id"].astype(str) == region]
            ax.scatter(
                g["x0"], g["y0"],
                s=1.5,
                alpha=0.45,
                color=REGION_COLORS[region],
                label=region,
            )
        ax.set_title(f"{dataset}: median-quadrant partition")
        ax.set_xlabel("x0")
        ax.set_ylabel("y0")
        ax.legend(frameon=False, markerscale=5, loc="best")
        add_panel_label(ax, label)

    # C-D: regional heatmaps
    def heatmap_for_dataset(ax, dataset, label):
        sub = regions[regions["dataset"].astype(str) == dataset].copy()
        sub["region_id"] = pd.Categorical(sub["region_id"], categories=REGION_ORDER, ordered=True)
        sub = sub.sort_values("region_id")

        mat = []
        for region in REGION_ORDER:
            r = sub[sub["region_id"] == region]
            mat.append([float(r[col].iloc[0]) for col, _ in PRIMARY_REGION_MEAN_COLS])
        mat = np.asarray(mat)

        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(f"{dataset}: regional bridge means")
        ax.set_xticks(np.arange(len(PRIMARY_REGION_MEAN_COLS)))
        ax.set_xticklabels([lab for _, lab in PRIMARY_REGION_MEAN_COLS], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(REGION_ORDER)))
        ax.set_yticklabels(REGION_ORDER)
        ax.set_ylabel("Region")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean bridge value")
        add_panel_label(ax, label)

    heatmap_for_dataset(axC, "CRC", "C")
    heatmap_for_dataset(axD, "Breast", "D")

    fig.suptitle(
        "Primary CRC and breast median-quadrant partitions and regional bridge heatmaps",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    out_png = OUTDIR / "Figure_S1_primary_partitions_and_bridge_heatmaps.png"
    out_pdf = OUTDIR / "Figure_S1_primary_partitions_and_bridge_heatmaps.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_pdf}")


# ==================================================
# Supplementary Figure S2
# ==================================================

def make_supplementary_figure_s2():
    shuffle_raw_csv = PRIMARY_VALIDATION / "within_section_shuffle_null_raw_long.csv"
    shuffle_summary_csv = PRIMARY_VALIDATION / "within_section_shuffle_null_summary.csv"
    het_csv = PRIMARY_VALIDATION / "within_section_heterogeneity_metrics_by_scheme.csv"

    for p in [shuffle_raw_csv, shuffle_summary_csv, het_csv]:
        require_file(p)

    raw = pd.read_csv(shuffle_raw_csv)
    summary = pd.read_csv(shuffle_summary_csv)
    het = pd.read_csv(het_csv)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    axA, axB, axC, axD = axes.flatten()

    # A-B: separate shuffle-null histograms
    def plot_shuffle(ax, dataset, label):
        metric = "mean_abs_pairwise_diff"
        vals = raw[
            (raw["dataset"].astype(str) == dataset)
            & (raw["feature"].astype(str) == "__overall__")
            & (raw["metric_name"].astype(str) == metric)
        ]["value"].dropna().to_numpy(float)

        row = summary[
            (summary["dataset"].astype(str) == dataset)
            & (summary["feature"].astype(str) == "__overall__")
            & (summary["metric_name"].astype(str) == metric)
        ].iloc[0]

        obs = float(row["observed_value"])
        null_mean = float(row["null_mean"])
        pval = float(row["empirical_p_upper"])

        ax.hist(vals, bins=40, color=DATASET_COLORS[dataset], alpha=0.75)
        ax.axvline(obs, color="black", linestyle="--", linewidth=2)
        ax.axvline(null_mean, color="#666666", linestyle=":", linewidth=1.5)

        ax.set_title(f"{dataset}: shuffle-null validation")
        ax.set_xlabel("Overall mean abs. inter-region difference")
        ax.set_ylabel("Permutation count")
        ax.text(
            0.62, 0.90,
            f"observed={obs:.4f}\nnull mean={null_mean:.4f}\nP={format_p(pval)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#dddddd"),
        )
        add_panel_label(ax, label)

    plot_shuffle(axA, "CRC", "A")
    plot_shuffle(axB, "Breast", "B")

    # C-D: partition sensitivity
    def get_values(dataset, metric):
        vals = []
        for scheme in SCHEME_ORDER:
            row = het[
                (het["dataset"].astype(str) == dataset)
                & (het["scheme"].astype(str) == scheme)
                & (het["feature"].astype(str) == "__overall__")
            ].iloc[0]
            vals.append(float(row[metric]))
        return vals

    for dataset in ["CRC", "Breast"]:
        axC.plot(
            SCHEME_LABELS,
            get_values(dataset, "region_range"),
            marker="o",
            linewidth=1.8,
            color=DATASET_COLORS[dataset],
            label=dataset,
        )
        axD.plot(
            SCHEME_LABELS,
            get_values(dataset, "mean_abs_pairwise_diff"),
            marker="o",
            linewidth=1.8,
            color=DATASET_COLORS[dataset],
            label=dataset,
        )

    axC.set_title("Partition sensitivity: regional range")
    axC.set_xlabel("Partition scheme")
    axC.set_ylabel("Value")
    axC.legend(frameon=False)
    add_panel_label(axC, "C")

    axD.set_title("Partition sensitivity: mean abs. inter-region difference")
    axD.set_xlabel("Partition scheme")
    axD.set_ylabel("Value")
    axD.legend(frameon=False)
    add_panel_label(axD, "D")

    fig.suptitle(
        "Primary CRC and breast within-section shuffle-null and partition-sensitivity analyses",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    out_png = OUTDIR / "Figure_S2_primary_shuffle_null_and_partition_sensitivity.png"
    out_pdf = OUTDIR / "Figure_S2_primary_shuffle_null_and_partition_sensitivity.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_pdf}")


# ==================================================
# Supplementary Figure S3
# ==================================================

def make_supplementary_figure_s3():
    region_csv = EXTERNAL_REGIONS / "external_region_program_means_by_scheme.csv"
    require_file(region_csv)

    df = pd.read_csv(region_csv)
    df = df[df["scheme"].astype(str) == "original"].copy()

    datasets = ["Lung cancer", "Prostate cancer", "Ovarian cancer"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))

    for ax, dataset, label in zip(axes, datasets, ["A", "B", "C"]):
        sub = df[df["dataset_label"].astype(str) == dataset].copy()

        mat = (
            sub.pivot_table(index="region_id", columns="program", values="mean_value", aggfunc="first")
            .reindex(REGION_ORDER)
            .reindex(columns=[p for p, _ in EXTERNAL_PROGRAMS])
        )

        im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="viridis")
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(EXTERNAL_PROGRAMS)))
        ax.set_xticklabels([lab for _, lab in EXTERNAL_PROGRAMS], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(REGION_ORDER)))
        ax.set_yticklabels(REGION_ORDER)
        ax.set_ylabel("Region")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Program score")
        add_panel_label(ax, label)

    fig.suptitle(
        "External solid-tumor region-program heatmaps",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_png = OUTDIR / "Figure_S3_external_solid_tumor_region_program_heatmaps.png"
    out_pdf = OUTDIR / "Figure_S3_external_solid_tumor_region_program_heatmaps.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_pdf}")


# ==================================================
# Supplementary Figure S4
# ==================================================

def make_supplementary_figure_s4():
    summary_csv = EXTERNAL_VALIDATION / "external_shuffle_null_summary.csv"
    het_csv = EXTERNAL_VALIDATION / "external_heterogeneity_metrics_by_scheme.csv"

    require_file(summary_csv)
    require_file(het_csv)

    summary = pd.read_csv(summary_csv)
    het = pd.read_csv(het_csv)

    datasets = ["Lung cancer", "Prostate cancer", "Ovarian cancer"]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    axA, axB, axC = axes

    # A: observed vs null summary
    sub = summary[
        (summary["program"].astype(str) == "__overall__")
        & (summary["metric_name"].astype(str) == "mean_abs_pairwise_diff")
    ].copy()
    sub["dataset_label"] = pd.Categorical(sub["dataset_label"], categories=datasets, ordered=True)
    sub = sub.sort_values("dataset_label")

    x = np.arange(len(sub))
    width = 0.34

    axA.bar(x - width / 2, sub["observed_value"], width, color="#4f79b8", label="Observed")
    axA.bar(x + width / 2, sub["null_mean"], width, color="#bbbbbb", label="Shuffle-null mean")

    axA.set_xticks(x)
    axA.set_xticklabels(datasets, rotation=0, ha="center")
    axA.set_ylabel("Overall mean abs. inter-region difference")
    axA.set_title("External shuffle-null validation")
    axA.legend(frameon=False)

    for i, row in enumerate(sub.itertuples(index=False)):
        axA.text(
            i - width / 2,
            row.observed_value,
            f"P={format_p(row.empirical_p_upper)}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    add_panel_label(axA, "A")

    # B-C: partition sensitivity
    def get_values(dataset, metric):
        vals = []
        for scheme in SCHEME_ORDER:
            row = het[
                (het["dataset_label"].astype(str) == dataset)
                & (het["scheme"].astype(str) == scheme)
                & (het["program"].astype(str) == "__overall__")
            ].iloc[0]
            vals.append(float(row[metric]))
        return vals

    for dataset in datasets:
        axB.plot(
            SCHEME_LABELS,
            get_values(dataset, "region_range"),
            marker="o",
            linewidth=1.8,
            label=dataset,
        )
        axC.plot(
            SCHEME_LABELS,
            get_values(dataset, "mean_abs_pairwise_diff"),
            marker="o",
            linewidth=1.8,
            label=dataset,
        )

    axB.set_title("Partition sensitivity: regional range")
    axB.set_xlabel("Partition scheme")
    axB.set_ylabel("Value")
    axB.legend(frameon=False, fontsize=8)
    add_panel_label(axB, "B")

    axC.set_title("Partition sensitivity: mean abs. inter-region difference")
    axC.set_xlabel("Partition scheme")
    axC.set_ylabel("Value")
    axC.legend(frameon=False, fontsize=8)
    add_panel_label(axC, "C")

    fig.suptitle(
        "External solid-tumor shuffle-null and partition-sensitivity summaries",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_png = OUTDIR / "Figure_S4_external_solid_tumor_shuffle_and_partition_sensitivity.png"
    out_pdf = OUTDIR / "Figure_S4_external_solid_tumor_shuffle_and_partition_sensitivity.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_pdf}")


def main():
    make_supplementary_figure_s1()
    make_supplementary_figure_s2()
    make_supplementary_figure_s3()
    make_supplementary_figure_s4()


if __name__ == "__main__":
    main()
