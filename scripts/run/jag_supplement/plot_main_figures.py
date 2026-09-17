#!/usr/bin/env python3
"""Publication figures for the R1 revision (Figures 3, 6, 7).

Hard-coded values are taken from JAG_FINAL_FIGURE_TABLE_DECISIONS.md
(Table 2 / Fig. 3 note, Supplement S11, Table 5). Matplotlib only.
"""
from __future__ import annotations
import os

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Hard-coded results (do not read logs at runtime)
# ---------------------------------------------------------------------------

# Fig. 3 — GF-FloodNet ResNet-50 test, seed 42.
# Optical-only / SAR-only: JAG_FINAL_FIGURE_TABLE_DECISIONS.md §2 Fig. 3 note.
# Optical+SAR (MA-XAttn): Table 2, ResNet-50 + ResNet-50, MA-XAttn row.
FIG3_MODELS = ["Optical", "SAR", "Dual (MA-XAttn)"]
# Macro-averaged (water + background) test metrics from the run logs:
#   optical-only  logs/gf_table2_fill/jag26_gf_optical_only_minep50_s42_523900.out
#   SAR-only      logs/gf_table2_fill/jag26_gf_sar_only_minep50_s42_523901.out
#   dual          logs/gf_table2_fill/jag26_gft2_resnet50_ma_s42_515380.out
FIG3_METRICS = ["Water IoU", "mIoU", "macro F1", "macro recall", "macro precision"]
FIG3_VALUES = {  # rows = models, cols = metrics (fractions)
    "Optical":         [0.9503, 0.9702, 0.9918, 0.9836, 0.9860],
    "SAR":             [0.8429, 0.9056, 0.9730, 0.9412, 0.9580],
    "Dual (MA-XAttn)": [0.9705, 0.9824, 0.9952, 0.9904, 0.9917],
}

# Fig. 6 — GF-FloodNet leave-one-region-out Water IoU (%).
# Source: JAG_FINAL_FIGURE_TABLE_DECISIONS.md §3 S11 (fifth column dropped).
# Horizontal reference 96.87: Table 2 ResNet-50 MA-XAttn Water IoU (in-distribution).
FIG6_REGIONS = [
    "Australia",
    "Brazil",
    "China",
    "India",
    "Pakistan",
    "Russia",
    "Mean",
]
FIG6_TICKS = [
    "Australia\n(1 scene, 31%)",
    "Brazil\n(1 scene, 17%)",
    "China\n(5 scenes, 14%)",
    "India\n(2 scenes, 19%)",
    "Pakistan\n(2 scenes, 8%)",
    "Russia\n(3 scenes, 11%)",
    "Mean",
]
FIG6_METHODS = [
    "Input concat",
    "Feature concat",
    "Gated fusion",
    "MA-XAttn",
]
# rows = regions (Australia … Mean); cols = FIG6_METHODS
FIG6_WATER_IOU = np.array(
    [
        [8.63, 9.99, 8.26, 11.49],
        [51.37, 56.98, 69.21, 62.43],
        [78.04, 65.42, 69.21, 67.95],
        [79.31, 84.44, 84.25, 80.57],
        [73.85, 75.28, 78.02, 76.19],
        [48.04, 73.87, 73.64, 78.52],
        [56.54, 61.00, 63.77, 62.86],
    ]
)
FIG6_INDIST_WATER_IOU = 97.05  # Table 2 ResNet-50 MA-XAttn (std.), seed 42

# Fig. 7 — S1S2-Water ResNet-50 test, mean ± SD over 3 seeds. Source: Table 5.
FIG7_METHODS = [
    "Input concatenation",
    "Feature addition",
    "Feature concatenation",
    "Cross-attention\n(bidirectional)",
    "Gated fusion",
    "MA-XAttn",
]
FIG7_PARAMS_M = [73.3, 96.8, 268.5, 102.4, 108.0, 147.1]
FIG7_FLOPS_G = [76.8, 87.2, 284.7, 88.3, 91.8, 102.0]
FIG7_WATER_IOU = [97.02, 96.91, 97.28, 97.06, 96.80, 97.30]  # MA-XAttn = standard protocol (Table 7), comparable with baselines
FIG7_SD = [0.29, 0.30, 0.06, 0.24, 0.37, 0.30]
FIG7_MA_OCC = (97.36, 0.06)  # occlusion-aware model (Table 6), open diamond for reference
FIG7_CONCAT_MEAN = 97.28
FIG7_CONCAT_SD = 0.06

DEFAULT_OUTDIR = Path(os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "paper/04_返修跟踪与写作/figures"))

# Okabe–Ito (colour-blind safe); skip yellow for bars/markers on white.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _thin_spines(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_fig3(outdir: Path) -> None:
    """GF-FloodNet modality contribution, grouped by metric (style of the submitted Fig. 3)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    n_metrics = len(FIG3_METRICS)
    x = np.arange(n_metrics)
    w = 0.26
    colors = ["#9ECAE1", "#FDAE6B", "#A1D99B"]  # pastel blue / orange / green
    for i, name in enumerate(FIG3_MODELS):
        vals = FIG3_VALUES[name]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colors[i], edgecolor="none", label=name, zorder=2)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="0.15", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(FIG3_METRICS)
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 1.005)
    ax.set_yticks([0.80, 0.84, 0.88, 0.92, 0.96, 1.00])
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("0.3")
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.6, borderaxespad=0.0)
    fig.tight_layout()
    _save(fig, outdir, "fig3_gf_modality_contribution")


def plot_fig6(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    n_reg = len(FIG6_REGIONS)
    n_m = len(FIG6_METHODS)
    x = np.arange(n_reg, dtype=float)
    w = 0.18
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * w
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
        OKABE_ITO["green"],
        OKABE_ITO["vermillion"],
    ]

    for j, (name, color) in enumerate(zip(FIG6_METHODS, colors)):
        ax.bar(
            x + offsets[j],
            FIG6_WATER_IOU[:, j],
            w,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=name,
            zorder=2,
        )

    # Mean group value labels only (other groups are too dense).
    mean_idx = n_reg - 1
    for j in range(n_m):
        val = FIG6_WATER_IOU[mean_idx, j]
        ax.text(
            x[mean_idx] + offsets[j],
            val + 1.6,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[j],
            rotation=90,
            zorder=3,
        )

    ax.axvline(mean_idx - 0.5, color="0.35", linestyle="--", linewidth=0.7, zorder=1)
    ax.axhline(
        FIG6_INDIST_WATER_IOU,
        color="0.35",
        linestyle="--",
        linewidth=0.7,
        zorder=1,
    )
    ax.text(
        x[-1] + 0.52,
        FIG6_INDIST_WATER_IOU + 1.2,
        "in-distribution test (Table 2)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.25",
        clip_on=False,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(FIG6_TICKS)
    ax.set_ylabel("Water IoU (%)")
    ax.set_xlim(-0.55, n_reg - 0.35)
    ax.set_ylim(0.0, 104.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75", zorder=0)
    ax.set_axisbelow(True)
    _thin_spines(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.38, 1.02),
        ncol=4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.1,
        handleheight=0.7,
        columnspacing=1.0,
        labelspacing=0.3,
    )
    fig.tight_layout()
    _save(fig, outdir, "fig6_gf_region_holdout")


def _flops_size(flops_g: float) -> float:
    # Marker area in pt^2; 80 G ≈ 36, 280 G ≈ 126.
    return float(flops_g) * 0.45


def plot_fig7(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    # Colours are described in the figure caption (no in-plot labels):
    # blue = input concatenation, light blue = feature addition, green = feature
    # concatenation, pink = cross-attention, grey = gated fusion, orange-red diamond = MA-XAttn.
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        "0.45",
        OKABE_ITO["vermillion"],
    ]
    is_ma = [name == "MA-XAttn" for name in FIG7_METHODS]

    lo = FIG7_CONCAT_MEAN - FIG7_CONCAT_SD
    hi = FIG7_CONCAT_MEAN + FIG7_CONCAT_SD
    ax.axhspan(lo, hi, facecolor=OKABE_ITO["green"], alpha=0.12, zorder=0, linewidth=0)
    ax.axhline(lo, color=OKABE_ITO["green"], linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)
    ax.axhline(hi, color=OKABE_ITO["green"], linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)

    for i, name in enumerate(FIG7_METHODS):
        marker = "D" if is_ma[i] else "o"
        z = 4 if is_ma[i] else 3
        ax.errorbar(
            FIG7_PARAMS_M[i],
            FIG7_WATER_IOU[i],
            yerr=FIG7_SD[i],
            fmt="none",
            ecolor=colors[i],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            zorder=2,
        )
        ax.scatter(
            FIG7_PARAMS_M[i],
            FIG7_WATER_IOU[i],
            s=_flops_size(FIG7_FLOPS_G[i]),
            marker=marker,
            facecolor=colors[i],
            edgecolor="0.15",
            linewidth=0.5,
            zorder=z,
        )

    i_ma = FIG7_METHODS.index("MA-XAttn")
    ax.errorbar(FIG7_PARAMS_M[i_ma] * 1.06, FIG7_MA_OCC[0], yerr=FIG7_MA_OCC[1], fmt="none", ecolor=colors[i_ma], elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=2)
    ax.scatter(FIG7_PARAMS_M[i_ma] * 1.06, FIG7_MA_OCC[0], s=_flops_size(FIG7_FLOPS_G[i_ma]), marker="D", facecolor="white", edgecolor=colors[i_ma], linewidth=1.0, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(60.0, 350.0)
    ax.set_ylim(96.2, 97.8)
    ax.set_xticks([70, 100, 150, 200, 300])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Params (M)")
    ax.set_ylabel("Water IoU (%)")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75", zorder=0)
    ax.set_axisbelow(True)
    _thin_spines(ax)

    size_handles = []
    for g in (80, 160, 280):
        h = ax.scatter(
            [],
            [],
            s=_flops_size(g),
            marker="o",
            facecolor="0.55",
            edgecolor="0.15",
            linewidth=0.4,
            label=f"{g} G",
        )
        size_handles.append(h)
    leg = ax.legend(
        handles=size_handles,
        title="FLOPs",
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        framealpha=0.95,
        borderpad=0.35,
        handletextpad=0.5,
        labelspacing=0.65,
        fontsize=8,
        title_fontsize=8,
    )
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, outdir, "fig7_s1s2_pareto")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot R1 revision main-text Figures 3, 6 and 7.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.expanduser().resolve()
    _apply_style()
    plot_fig3(outdir)
    plot_fig6(outdir)
    plot_fig7(outdir)
    for stem in (
        "fig3_gf_modality_contribution",
        "fig6_gf_region_holdout",
        "fig7_s1s2_pareto",
    ):
        for ext in ("pdf", "png"):
            path = outdir / f"{stem}.{ext}"
            print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
