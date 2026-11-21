"""Generate slide charts from verified report tables (English labels).

All numbers are copied from Project03_Report.pdf tables:
  - Table VII  (Visual Genome retrieval, 3 seeds)
  - Table VIII (Visual Genome ablation, 1 run each)
  - Table X    (DB15K / FB15K-237-IMG test, 3 seeds)
  - Table XI   (graph-structure ablation, 1 run each)
  - Table XIII (ViHeritage-MMKG test, 3 seeds for full models)
  - Table XIV  (XAI fidelity)
  - report/figures/scripts/make_figures.py (gain-kept series, Figure 7)

Run: python3 slide/make_slide_charts.py
Output: slide/assets/charts/*.png
No number is recomputed on test; gain-kept percentages reuse the report script.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY = "#165A9F", "#EB6834", "#1BAF7A", "#8A94A6"
INK, INK2, GRID = "#1A1A1A", "#5B6570", "#E4E3DF"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar_label(ax, x, y, v, fmt="{:.4f}", dy=0.0):
    ax.text(x, y + dy, fmt.format(v).replace("-", "\u2212"), ha="center", va="bottom",
            fontsize=8, color=INK)


# --- Chart 1: Level-1 retrieval, Table VII (3 seeds for Full Graph) ---------
def chart_l1_retrieval():
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    clip = [0.0295, 0.0793, 0.1165, 0.0611]
    full = [0.0363, 0.0961, 0.1381, 0.0718]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    b1 = ax.bar(x - w / 2, clip, w, color=GRAY, label="CLIP-only", edgecolor="white")
    b2 = ax.bar(x + w / 2, full, w, color=BLUE, label="CLIP + GAT (Full Graph, 3 seeds)",
                edgecolor="white")
    for bars in (b1, b2):
        for r in bars:
            bar_label(ax, r.get_x() + r.get_width() / 2, r.get_height(), r.get_height())
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 0.165)
    ax.set_ylabel("Test score (Visual Genome)")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    ax.grid(axis="x", visible=False)
    save(fig, "l1_retrieval")


# --- Chart 2: Level-1 ablation, Table VIII (1 run each) ---------------------
def chart_l1_ablation():
    names = ["CLIP-only", "No-edge", "Random-edges", "Full Graph"]
    mrr = [0.0611, 0.0692, 0.0708, 0.0717]
    kept = ["baseline", "77% of gain", "92% of gain", "100%"]
    colors = [GRAY, AQUA, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    bars = ax.bar(names, mrr, color=colors, edgecolor="white", width=0.62)
    for r, k in zip(bars, kept):
        bar_label(ax, r.get_x() + r.get_width() / 2, r.get_height(), r.get_height())
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() / 2, k, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_ylabel("Test MRR (Visual Genome, 1 run each)")
    ax.set_ylim(0, 0.082)
    ax.grid(axis="x", visible=False)
    save(fig, "l1_ablation")


# --- Chart 3: Level-2 direct vs indirect, Table X (3 seeds) -----------------
def chart_l2_direct_indirect():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=False)
    panels = [
        ("DB15K", ["Direct\n(2 282 queries)", "Indirect\n(3 195 queries)"],
         [[0.5520, 0.0475], [0.5520, 0.3143]]),
        ("FB15K-237-IMG", ["Direct\n(8 592 queries)", "Indirect\n(24 564 queries)"],
         [[0.3475, 0.0122], [0.7770, 0.1186]]),
    ]
    for ax, (title, labels, vals) in zip(axes, panels):
        x = np.arange(2)
        w = 0.36
        clip = [vals[0][0], vals[0][1]]
        adamf = [vals[1][0], vals[1][1]]
        b1 = ax.bar(x - w / 2, clip, w, color=GRAY, label="CLIP", edgecolor="white")
        b2 = ax.bar(x + w / 2, adamf, w, color=BLUE, label="CLIP + AdaMF", edgecolor="white")
        for bars in (b1, b2):
            for r in bars:
                bar_label(ax, r.get_x() + r.get_width() / 2, r.get_height(), r.get_height())
        ax.set_xticks(x, labels)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 0.92)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test MRR (3 seeds)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2,
               frameon=False)
    fig.tight_layout(w_pad=1.6)
    save(fig, "l2_direct_indirect")


# --- Chart 4: cross-level gain-kept (report Figure 7 data, English) --------
def gain(v, clip, full):
    return 100.0 * (v - clip) / (full - clip)


def chart_gain_kept():
    # Same series as report/figures/scripts/make_figures.py (Figure 7)
    ablation = [
        ("Visual Genome, GAT", gain(0.0691928, 0.0611489, 0.0716555),
         gain(0.0707720, 0.0611489, 0.0716555), None),
        ("DB15K, AdaMF", 0.0, 36.9, 0.0),
        ("FB15K-237, AdaMF", 0.0, 93.1, 92.9),
        ("ViHeritage, AdaMF", 0.0, -29.8, None),
        ("ViHeritage, GAT", -1.3, -3.0, None),
    ]
    series = [("Remove edges", BLUE, 1), ("Shuffle edges, keep degree", ORANGE, 2),
              ("Random edges", AQUA, 3)]
    y = np.arange(len(ablation))[::-1] * 1.0
    h = 0.25
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    for k, (name, color, idx) in enumerate(series):
        off = (1 - k) * (h + 0.03)
        first = True
        for yi, row in zip(y, ablation):
            v = row[idx]
            if v is None:
                continue
            ax.barh(yi + off, v, height=h, color=color, edgecolor="white", linewidth=0.8,
                    label=name if first else None)
            first = False
            if abs(v) >= 1.5:
                txt = f"{v:.0f}%".replace("-", "\u2212")
                ax.text(v + (2 if v >= 0 else -2), yi + off, txt, va="center",
                        ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(100, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(100, y[0] + 0.55, "full", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks(y, [r[0] for r in ablation])
    ax.set_xlim(-55, 112)
    ax.set_xlabel("Gain over CLIP retained (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=0.9)
    save(fig, "gain_kept")


# --- Chart 5: XAI fidelity, Table XIV / Figure 4 data -----------------------
def chart_xai_fidelity():
    xai = [
        ("VG\nGAT", 0.118, 0.002, 0.003, 0.0004),
        ("DB15K\nAdaMF", 0.511, 0.065, 0.023, 0.011),
        ("FB15K\nAdaMF", 0.335, 0.041, 0.011, 0.044),
        ("ViHeritage\nAdaMF", 0.535, 0.059, 0.145, 0.032),
        ("ViHeritage\nGAT", 0.103, 0.080, 0.002, 0.014),
    ]
    x = np.arange(len(xai))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for k, (name, color, i, j) in enumerate([
            ("Remove explanation set S", BLUE, 1, 2),
            ("Remove same # random elements", ORANGE, 3, 4)]):
        vals = [r[i] for r in xai]
        sds = [r[j] for r in xai]
        pos = x + (k - 0.5) * (w + 0.03)
        ax.bar(pos, vals, width=w, color=color, edgecolor="white", linewidth=0.8, label=name,
               yerr=sds, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK2})
        if k == 0:
            for p, v, s in zip(pos, vals, sds):
                bar_label(ax, p, v + s + 0.015, v, fmt="{:.3f}")
    ax.set_xticks(x, [r[0] for r in xai])
    ax.set_ylabel("Fid+ (higher is better)")
    ax.set_ylim(0, 0.68)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2,
              handlelength=1.0, columnspacing=1.0)
    save(fig, "xai_fidelity")


# --- Chart 6: ViHeritage, Table XIII (Vietnamese queries) -------------------
def chart_viheritage():
    names = ["CLIP", "CLIP + AdaMF", "CLIP + GAT"]
    pooled = [0.139, 0.180, 0.195]
    pooled_err = [0.0, 0.004, 0.014]
    indirect = [0.112, 0.219, 0.258]
    indirect_err = [0.0, 0.012, 0.036]
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, pooled, w, color=GRAY, yerr=pooled_err, capsize=3,
                label="All 142 test queries", edgecolor="white",
                error_kw={"elinewidth": 0.8, "ecolor": INK2})
    b2 = ax.bar(x + w / 2, indirect, w, color=BLUE, yerr=indirect_err, capsize=3,
                label="Indirect queries (54)", edgecolor="white",
                error_kw={"elinewidth": 0.8, "ecolor": INK2})
    for bars in (b1, b2):
        for r in bars:
            bar_label(ax, r.get_x() + r.get_width() / 2, r.get_height(), r.get_height(),
                      fmt="{:.3f}")
    ax.set_xticks(x, names)
    ax.set_ylabel("Test MRR (ViHeritage-MMKG, vi)")
    ax.set_ylim(0, 0.33)
    ax.legend(loc="upper left", frameon=False, framealpha=0.9)
    ax.grid(axis="x", visible=False)
    save(fig, "viheritage")


if __name__ == "__main__":
    chart_l1_retrieval()
    chart_l1_ablation()
    chart_l2_direct_indirect()
    chart_gain_kept()
    chart_xai_fidelity()
    chart_viheritage()
    print("Charts written to", OUT)
