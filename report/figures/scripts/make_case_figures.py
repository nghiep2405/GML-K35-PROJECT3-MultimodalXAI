"""Vẽ hình ca giải thích cho cả ba cấp cùng một kiểu (src/notebooks/vg_xai_plot.py): ảnh | đồ thị giải thích.
Nút cam = ảnh hoặc thực thể của ảnh được trả về, nút xanh ngọc = thực thể neo của truy vấn. Cạnh xanh dương đậm = S,
độ dày và số trong ngoặc = Δ điểm đồ thị khi bỏ phần tử đó; cạnh xanh nhạt mảnh = bộ ba nối S về thực thể neo;
nét đứt = quan hệ được hỏi.

Chạy: python report/figures/scripts/make_case_figures.py
Số liệu chép nguyên từ kết quả đã khóa trên Drive:
  Visual Genome  results/VisualGenome/XAI/V6_cases.md, ca vg:text:575905 (seed 2026, K* = 12; Δ làm tròn 10^-4)
  DB15K          results/DB15K/.../C8_case_analysis.md, ca q_ffbbb7ed71136914b34d1caf (seed 2027, K* = 2)
  ViHeritage     results/MMKG/.../D4_cases.md, ca gat_02 (seed 2027, K* = 8)
Ảnh: case_inputs/vg_2403818_from_notebook11.png (cắt từ hình ca của notebook 11, ảnh Visual Genome 2403818);
     case_inputs/db15k_hitchens_from_C8.png (cắt từ hình ca gốc của notebook 07);
     case_inputs/viheritage_img0185.jpg (data/MMKG/images/img0185.jpg).
"""
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
sys.path.insert(0, str(FIG.parents[1] / "src" / "notebooks"))
from vg_xai_plot import (C_ANCHOR, C_EDGE, C_IMG, C_LINK, C_OTHER, C_QUERY, FIG_H, FIG_W, FS, ROOT,  # noqa: E402
                         _layout, case_figure_items)


def wrap(s, w=14):
    return "\n".join(textwrap.wrap(s, w))


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- cấp 1
def vg_peace_sign():
    # (đầu, quan hệ, đuôi), số quan hệ cùng tên được gộp, tổng Δ (V6_cases.md)
    items = [((ROOT, "contains", "peace sign"), 5, 5 * 0.0055),
             (("man", "wearing", "suit"), 5, 3 * 0.0003 + 2 * 0.0002),
             (("man", "holding up", "peace sign"), 2, 2 * 0.0002)]
    img = Image.open(FIG / "case_inputs" / "vg_2403818_from_notebook11.png")
    fig = case_figure_items([("Ảnh đúng, hạng 1\nbỏ giải thích S: hạng 4", img)], items, 12,
                            "[thanh_cong] Guy holding up peace sign\nCLIP hạng 15 → CLIP + GAT hạng 1")
    save(fig, "case_vg_success_peacesign.png")


# ----------------------------------------------------------------------------- cấp 2 và 3
def draw_kg(ax, cand, anchor, facts, links, query, title, pos=None, label_width=14, label_side=None, pad=None):
    """facts: (đầu, quan hệ, đuôi, Δ) thuộc tập giải thích S; links: bộ ba train nối S về thực thể neo (không thuộc S);
    query: (đầu, quan hệ, đuôi) của quan hệ được hỏi, vẽ nét đứt. pos: vị trí nút cố định (tùy chọn).
    label_side: {nút: 'left'|'right'|'above'|'below'} để ghi tên nút ra ngoài vòng tròn; pad: (trái, phải, dưới, trên)."""
    G = nx.MultiDiGraph()
    dmax = max(d for *_, d in facts)
    for h, r, t, d in facts:
        G.add_edge(h, t, label=f"{r}\n({d:.3f})", kind="fact", width=1.0 + 3.0 * d / dmax)  # hai dòng cho nhãn ngắn
    for h, r, t in links:
        G.add_edge(h, t, label=r, kind="link", width=0.9)
    G.add_edge(query[0], query[2], label=query[1] + " ?", kind="query", width=1.2)
    pos = pos or _layout(nx.DiGraph(G))
    nodes = list(G)
    nsize = {x: 1000 if x in (cand, anchor) else 420 for x in nodes}
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_size=[nsize[x] for x in nodes],
                           node_color=[C_IMG if x == cand else C_ANCHOR if x == anchor else C_OTHER for x in nodes])
    # Hai cạnh cùng cặp nút (kể cả ngược chiều) được uốn cong khác nhau để cạnh và nhãn không đè nhau.
    seen = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        key = frozenset((u, v))
        j = seen.get(key, 0)
        seen[key] = j + 1
        # cặp ngược chiều: cả hai cùng uốn rad 0.4 (theo chiều riêng) nên tách về hai phía đối xứng
        two_way = G.has_edge(v, u) and d["kind"] != "query"
        cs = f"arc3,rad={0.4 if two_way else [0.0, 0.3, -0.3, 0.55][j]}"
        style, color = dict(fact=("-", C_EDGE), link=("-", C_LINK), query=("--", C_QUERY))[d["kind"]]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], style=style, width=d["width"], edge_color=color,
                               arrows=True, arrowsize=12, nodelist=nodes, node_size=[nsize[x] for x in nodes],
                               connectionstyle=cs)
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={(u, v): d["label"]}, font_size=FS["edge"],
                                     connectionstyle=cs, bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))
    label_side = label_side or {}
    center = [x for x in nodes if x not in label_side]
    nx.draw_networkx_labels(G, pos, ax=ax, labels={x: wrap(x, label_width) for x in center}, font_size=FS["node"])
    off = dict(left=((-15, 0), "right", "center"), right=((15, 0), "left", "center"),
               above=((0, 15), "center", "bottom"), below=((0, -15), "center", "top"))
    for x, side in label_side.items():
        (dx, dy), ha, va = off[side]
        ax.annotate(wrap(x, label_width), pos[x], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=FS["node"], zorder=5)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    l, r, b, t = pad or (0.5, 0.5, 0.3, 0.3)
    ax.set_xlim(min(xs) - l, max(xs) + r)
    ax.set_ylim(min(ys) - b, max(ys) + t)
    ax.axis("off")
    ax.set_title(title, fontsize=FS["title"])


def figure(image, image_title, suptitle, height=FIG_H, graph_ratio=1.6):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, height), gridspec_kw={"width_ratios": [1.0, graph_ratio]})
    axes[0].imshow(image)
    axes[0].set_title(image_title, fontsize=FS["title"])
    axes[0].axis("off")
    fig.suptitle(suptitle, fontsize=FS["sup"])
    return fig, axes


def db15k_hitchens():
    img = Image.open(FIG / "case_inputs" / "db15k_hitchens_from_C8.png")
    fig, axes = figure(img, "Ảnh đúng: Christopher Hitchens\nhạng 1",
                       "[thanh_cong] Find an image of the entity whose influenced by relation points to\n"
                       "Richard Dawkins. CLIP hạng 55 → CLIP + AdaMF hạng 1")
    draw_kg(axes[1], "Christopher Hitchens", "Richard Dawkins",
            facts=[("Christopher Hitchens", "influenced", "Richard Dawkins", 0.1982),
                   ("Christopher Hitchens", "influenced by", "Albert Camus", 0.1291)],
            links=[], query=("Christopher Hitchens", "influenced by", "Richard Dawkins"),
            title="Giải thích S (K* = 2); số trong ngoặc: Δ điểm đồ thị",
            pos={"Christopher Hitchens": (1.0, 1.0), "Richard Dawkins": (2.0, -0.6), "Albert Camus": (0.0, -0.6)},
            label_side={"Christopher Hitchens": "above", "Richard Dawkins": "below", "Albert Camus": "below"},
            pad=(0.6, 0.6, 0.6, 0.5))
    fig.tight_layout()
    save(fig, "case_db15k_success_hitchens.png")


def viheritage_hue():
    img = Image.open(FIG / "case_inputs" / "viheritage_img0185.jpg")
    fig, axes = figure(img, "Ảnh hạng 1 (đúng):\nChùa Thiên Mụ",
                       "[thanh_cong] Thành phố Huế có những di tích nào?\nCLIP + GAT: ảnh đúng hạng 1",
                       height=4.8, graph_ratio=2.4)
    facts = [("Chùa Thiên Mụ", "thuộc quần thể", "Quần thể di tích Cố đô Huế", 0.082),
             ("Hoàng thành Huế", "nằm ở tỉnh", "Huế", 0.081),
             ("Nhà thờ chính tòa Phủ Cam", "nằm ở tỉnh", "Huế", 0.061),
             ("Lăng Tự Đức", "nằm ở tỉnh", "Huế", 0.038),
             ("Lăng Khải Định", "nằm ở tỉnh", "Huế", 0.032),
             ("Lăng Minh Mạng", "nằm ở tỉnh", "Huế", 0.030),
             ("Chùa Thiên Mụ", "ở gần", "Lăng Tự Đức", 0.015),
             ("Lăng Tự Đức", "ở gần", "Chùa Thiên Mụ", 0.014)]
    # Bộ ba nối bộ ba giải thích đứng đầu về thực thể neo (đường đi hai bước Huế → Quần thể → Chùa Thiên Mụ).
    links = [("Quần thể di tích Cố đô Huế", "nằm ở tỉnh", "Huế")]
    # Thực thể của ảnh bên trái, thực thể neo ở giữa, bốn di tích chỉ nối với neo xếp hình nan quạt bên phải
    # (các cạnh dài bằng nhau, tách góc đều để nhãn không chồng).
    import math
    hue = (3.4, 0.0)
    fan = {"Hoàng thành Huế": 55, "Nhà thờ chính tòa Phủ Cam": 18, "Lăng Khải Định": -18, "Lăng Minh Mạng": -55}
    pos = {"Chùa Thiên Mụ": (0.0, 0.0), "Quần thể di tích Cố đô Huế": (1.7, 2.0), "Lăng Tự Đức": (1.7, -2.0),
           "Huế": hue}
    pos.update({k: (hue[0] + 2.9 * math.cos(math.radians(a)), 1.45 * 2.9 * math.sin(math.radians(a)) / 1.45)
                for k, a in fan.items()})
    side = {"Chùa Thiên Mụ": "left", "Quần thể di tích Cố đô Huế": "above", "Lăng Tự Đức": "below",
            "Hoàng thành Huế": "right", "Nhà thờ chính tòa Phủ Cam": "right", "Lăng Khải Định": "right",
            "Lăng Minh Mạng": "right"}
    draw_kg(axes[1], "Chùa Thiên Mụ", "Huế", facts, links, ("Chùa Thiên Mụ", "nằm ở tỉnh", "Huế"),
            "Giải thích S (K* = 8); số trong ngoặc: Δ điểm đồ thị", pos=pos, label_width=14, label_side=side,
            pad=(1.6, 2.1, 0.75, 0.75))
    fig.tight_layout()
    save(fig, "case_viheritage_gat_hue.png")


if __name__ == "__main__":
    vg_peace_sign()
    db15k_hitchens()
    viheritage_hue()
    print("Đã ghi hình ca vào", FIG)
