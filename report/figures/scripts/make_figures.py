"""Vẽ lại các hình định lượng cho báo cáo Project 03 từ kết quả đã khóa trong `0. final`.

Chạy: python report/figures/scripts/make_figures.py
Đầu ra: report/figures/fig_*.pdf (vector, dùng trong LaTeX) và .png (xem nhanh).

Mọi số liệu dưới đây chép nguyên từ các file kết quả trên Drive (`MLGs - Project/0. final/results/...`);
đường dẫn nguồn ghi ở từng khối. Không có số nào được tính lại trên test.
Quy ước trình bày: mỗi hình một thông điệp; không dùng dấu gạch ngang dài trong chữ trên hình.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

# Bảng màu phân loại (3 slot đầu của palette mặc định, đã kiểm tra CVD all-pairs).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.unicode_minus": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

COL_W = 3.45   # một cột IEEE (inch)
PAGE_W = 7.16  # hai cột


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def bar_label(ax, x, y, v, horizontal=False):
    txt = f"{v:.0f}%" if horizontal else f"{v:.2f}"
    txt = txt.replace("-", "−")  # dấu trừ toán học, không phải gạch ngang
    if horizontal:
        ax.text(v + (2 if v >= 0 else -2), y, txt, va="center", ha="left" if v >= 0 else "right",
                fontsize=6.5, color=INK)
    else:
        ax.text(x, y, txt, ha="center", va="bottom", fontsize=6.5, color=INK)


# ---------------------------------------------------------------------------
# Hình 1. Thông điệp: cạnh thật quan trọng ở đâu.
# gain_kept = (MRR biến thể - MRR CLIP) / (MRR mô hình đầy đủ - MRR CLIP), seed ablation.
# Nguồn:
#  Visual Genome results/VisualGenome/final_test_summary.json (MRR toàn tập, seed 2026)
#  DB15K         results/DB15K/ABLATION/README_ABLATION.md (MRR gián tiếp, seed 2027)
#  FB15K-237     results/FB15K/ABLATION/README_ABLATION.md (MRR gián tiếp, seed 2027)
#  ViHeritage    results/MMKG/test_locked_20261117T040900Z/D1_ablation.csv (MRR gián tiếp, vi, seed 2027)
# Visual Genome random_edges hoán vị đỉnh đích, giữ bậc, nên được xếp vào nhóm "xáo cạnh".
# ---------------------------------------------------------------------------
def gain(v, clip, full):
    return 100.0 * (v - clip) / (full - clip)


ablation = [
    # (nhãn, bỏ cạnh, xáo cạnh giữ bậc, cạnh ngẫu nhiên)
    ("Visual Genome, GAT", gain(0.0691928, 0.0611489, 0.0716555), gain(0.0707720, 0.0611489, 0.0716555), None),
    ("DB15K, AdaMF", 0.0, 36.9, 0.0),
    ("FB15K-237, AdaMF", 0.0, 93.1, 92.9),
    ("ViHeritage, AdaMF", 0.0, -29.8, None),
    ("ViHeritage, GAT", -1.3, -3.0, None),
]


def fig_ablation():
    series = [("Bỏ cạnh", BLUE, 1), ("Xáo cạnh, giữ bậc", ORANGE, 2), ("Cạnh ngẫu nhiên", AQUA, 3)]
    y = np.arange(len(ablation))[::-1] * 1.0
    h = 0.25
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
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
            if abs(v) >= 1.5:  # cột bằng 0 không ghi nhãn để tránh rối; ý nghĩa nêu trong chú thích
                bar_label(ax, None, yi + off, v, horizontal=True)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(100, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(100, y[0] + 0.55, "đầy đủ", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in ablation])
    ax.set_xlim(-55, 112)
    ax.set_xlabel("Lợi ích so với CLIP còn giữ lại (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=0.9)
    save(fig, "fig_ablation_gain_kept")


# ---------------------------------------------------------------------------
# Hình 2. Thông điệp: bỏ đúng tập giải thích S làm ảnh trả về mất điểm nhiều hơn hẳn bỏ ngẫu nhiên.
# Nguồn: results/XAI_L3_all_levels.md và results/VisualGenome/XAI/XAI_L3_row.csv (trung bình ± SD qua 3 seed).
# PN, PS, heuristic để trong bảng.
# ---------------------------------------------------------------------------
xai = [
    # nhãn, Fid+ S, sd, Fid+ ngẫu nhiên, sd
    ("VG\nGAT", 0.118, 0.002, 0.003, 0.0004),
    ("DB15K\nAdaMF", 0.511, 0.065, 0.023, 0.011),
    ("FB15K\nAdaMF", 0.335, 0.041, 0.011, 0.044),
    ("ViHeritage\nAdaMF", 0.535, 0.059, 0.145, 0.032),
    ("ViHeritage\nGAT", 0.103, 0.080, 0.002, 0.014),
]


def fig_fidelity():
    x = np.arange(len(xai))
    w = 0.36
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for k, (name, color, i, j) in enumerate([("Bỏ tập giải thích S", BLUE, 1, 2),
                                             ("Bỏ cùng số phần tử ngẫu nhiên", ORANGE, 3, 4)]):
        vals = [r[i] for r in xai]
        sds = [r[j] for r in xai]
        pos = x + (k - 0.5) * (w + 0.03)
        ax.bar(pos, vals, width=w, color=color, edgecolor="white", linewidth=0.8, label=name,
               yerr=sds, error_kw={"elinewidth": 0.7, "capsize": 1.8, "ecolor": INK2})
        if k == 0:
            for p, v, s in zip(pos, vals, sds):
                bar_label(ax, p, v + s + 0.015, v)
    ax.set_xticks(x, [r[0] for r in xai], fontsize=6.5)
    ax.set_ylabel("Fid+ (cao hơn là tốt hơn)")
    ax.set_ylim(0, 0.68)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=2,
              handlelength=1.0, columnspacing=1.0)
    save(fig, "fig_xai_fidelity")


# ---------------------------------------------------------------------------
# Hình 3. Thông điệp: K* (chọn trên validation) là điểm đánh đổi giữa độ trung thực và độ gọn.
# Nguồn: results/{DB15K,FB15K,MMKG}/XAI_L3/D5_sparsity_K_curve.csv (seed chính);
#        results/VisualGenome/XAI/V4_sparsity_K_curve.csv (seed chính 2026).
# ---------------------------------------------------------------------------
kcurve = {
    "Visual Genome, GAT": dict(K=[1, 2, 3, 5, 8, 12, 16], kstar=12,
                               fid=[0.025, 0.042, 0.057, 0.078, 0.100, 0.117, 0.125],
                               sp=[0.964, 0.927, 0.893, 0.828, 0.742, 0.649, 0.579]),
    "DB15K, AdaMF": dict(K=[1, 2, 3, 5, 8], kstar=2,
                         fid=[0.4315, 0.5362, 0.5882, 0.6577, 0.7016],
                         sp=[0.871, 0.801, 0.745, 0.661, 0.595]),
    "FB15K-237, AdaMF": dict(K=[1, 2, 3, 5, 8], kstar=8,
                             fid=[0.1007, 0.1687, 0.2134, 0.2842, 0.3501],
                             sp=[0.967, 0.934, 0.910, 0.869, 0.813]),
    "ViHeritage, AdaMF": dict(K=[1, 2, 3, 5, 8], kstar=1,
                              fid=[0.4845, 0.4730, 0.3660, 0.4023, 0.4028],
                              sp=[0.759, 0.596, 0.497, 0.425, 0.408]),
    "ViHeritage, GAT": dict(K=[1, 2, 3, 5, 8], kstar=8,
                            fid=[0.0721, 0.1025, 0.1228, 0.1437, 0.1495],
                            sp=[0.964, 0.928, 0.894, 0.828, 0.761]),
}


def fig_kcurve():
    fig, axes = plt.subplots(1, len(kcurve), figsize=(PAGE_W, 1.8), sharey=True)
    for ax, (title, d) in zip(axes, kcurve.items()):
        ax.plot(d["K"], d["sp"], color=AQUA, lw=1.6, marker="^", ms=3.2, label="Sparsity")
        ax.plot(d["K"], d["fid"], color=BLUE, lw=1.6, marker="o", ms=3.2, label="Fid+")
        i = d["K"].index(d["kstar"])
        ax.plot([d["kstar"]] * 2, [d["fid"][i], d["sp"][i]], color=INK2, lw=0.8, linestyle=(0, (2, 2)))
        ha = "left" if d["kstar"] == min(d["K"]) else ("right" if d["kstar"] == max(d["K"]) else "center")
        ax.annotate(f"K*={d['kstar']}", (d["kstar"], d["sp"][i]), xytext=(0, 5), textcoords="offset points",
                    ha=ha, fontsize=6.5, color=INK2)
        ax.set_title(title, fontsize=7.2)
        ax.set_xticks([k for k in d["K"] if k in (1, 2, 3, 5, 8, 12, 16) and not (k == 2 and max(d["K"]) > 8)])
        ax.set_xlabel("K")
        ax.set_ylim(0, 1.08)
    fig.tight_layout(w_pad=0.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, handlelength=1.6)
    save(fig, "fig_xai_k_curve")


if __name__ == "__main__":
    fig_ablation()
    fig_fidelity()
    fig_kcurve()
    print("Đã ghi hình vào", OUT)
