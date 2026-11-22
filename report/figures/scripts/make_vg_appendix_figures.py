"""Hình phụ lục cho 9 ca còn lại của Bảng VII (Visual Genome): đúng kiểu Hình 2 (case_vg_success_peacesign.png).

Cách vẽ chép nguyên src/notebooks/vg_xai_plot.draw_graph và case_figure_items (màu, cỡ nút, độ dày cạnh, nhãn
'chứa (Δ)', thang 10^expo, lề trục, kích thước hình 8.0 × 3.6 inch, tiêu đề); chỉ thay spring layout bằng vị trí nút đặt tay
cho từng hình để nhãn không chồng nhau. Ca thất bại vẽ ảnh đúng và tập giải thích S của ảnh đúng, như cột "Phần tử đóng góp
lớn nhất" của Bảng VII; ảnh sai đứng hạng 1 được nêu trong chú thích hình.

Số liệu: case_inputs/appendix_vg/V6_cases.md (chép nguyên results/VisualGenome/XAI/V6_cases.md, checkpoint chính, K* = 12).
Ảnh: case_inputs/appendix_vg/vg_<mã>_{top,gold}.jpg, cắt từ hình ca của notebook 11 (prepare_appendix_inputs).
Chạy: python report/figures/scripts/make_vg_appendix_figures.py   ->   report/figures/appendix_vg/vg_<mã>.png
"""
import math
import re
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
INP = FIG / "case_inputs" / "appendix_vg"
OUT = FIG / "appendix_vg"
sys.path.insert(0, str(FIG.parents[1] / "src" / "notebooks"))
from vg_xai_plot import C_EDGE, C_IMG, C_OTHER, FIG_H, FIG_W, FS, ROOT, _show_image  # noqa: E402


# ---------------------------------------------------------------------------- số liệu
def read_vg(md):
    t = Path(md).read_text(encoding="utf-8")
    cases = []
    for blk in t.split("\n## ")[1:]:
        loai, cid = re.match(r"(\w+): vg:text:(\d+)", blk).groups()
        q = re.search(r"\*\*Truy vấn:\*\* (.*?)\s*$", blk, re.M).group(1).strip()
        r = re.search(r"CLIP (\d+) → CLIP \+ GAT (\d+) · no_edge (\d+) · random_edges (\d+)", blk).groups()
        imgs = {}
        for part in re.split(r"\n(?=\*\*Ảnh)", blk)[1:]:
            kind = "gold" if part.startswith("**Ảnh đúng") else "top"
            m = re.search(r"hạng (\d+), bỏ giải thích → hạng (\d+)", part)
            rows = re.findall(r"^\| \d+ \| (.+?) \| ([-+.\de]+) \|$", part, re.M)
            imgs[kind] = dict(rank=int(m.group(1)), rank_rm=int(m.group(2)), S=[(s, float(d)) for s, d in rows])
        cases.append(dict(loai=loai, id=cid, query=q, clip=int(r[0]), model=int(r[1]), **imgs))
    return cases


VG_RELS = sorted(["on top of", "next to", "hanging out", "falling to", "worn by", "holding up", "quan hệ lạ", "wearing",
                  "saying", "display", "behind", "above", "have", "has", "with", "and", "in", "on", "of", "by"],
                 key=len, reverse=True)


def vg_triple(sentence):
    if sentence.startswith("ảnh chứa "):
        return ROOT, "contains", sentence[len("ảnh chứa "):]
    for r in VG_RELS:
        m = re.match(rf"^(.+?) {re.escape(r)} (.+)$", sentence)
        if m:
            return m.group(1), r, m.group(2)
    raise ValueError(sentence)


def vg_items(S):
    """Như vg_xai_plot.merged_explanation: gộp quan hệ trùng (đầu, quan hệ, đuôi), cộng Δ, sắp Δ giảm dần."""
    groups = OrderedDict()
    for s, d in S:
        k = vg_triple(s)
        n, tot = groups.get(k, (0, 0.0))
        groups[k] = (n + 1, tot + d)
    return sorted(((k, n, tot) for k, (n, tot) in groups.items()), key=lambda x: -x[2])


# ---------------------------------------------------------------------------- vẽ (chép draw_graph, pos đặt tay)
def draw_graph_pos(ax, items, k_star, pos):
    ax.axis("off")
    G = nx.DiGraph()
    G.add_node(ROOT)
    size, mult, loops = {}, {}, {}
    expo = int(math.floor(math.log10(max(t for _, _, t in items))))
    for (s, r, d), n, t in items:
        for x in (s, d):
            G.add_node(x)
            if x != ROOT:
                size[x] = size.get(x, 0.0) + t
        if s == ROOT:
            mult[d] = max(mult.get(d, 1), n)
        lab = (("chứa" if r == "contains" else r) + (f" ×{n}" if n > 1 and s != ROOT else "")
               + f" ({t / 10 ** expo:.3g})")
        if s == d:
            loops.setdefault(s, []).append(lab)
            continue
        if G.has_edge(s, d):
            G.edges[s, d]["label"] += " / " + lab
            G.edges[s, d]["delta"] += t
        else:
            G.add_edge(s, d, label=lab, delta=t)
    for e in G.edges:
        G.edges[e]["label"] = textwrap.shorten(G.edges[e]["label"], 40, placeholder="...")
    missing = set(G) - set(pos)
    assert not missing, f"thiếu vị trí cho {missing}"
    vmax = max(size.values()) if size else 1.0
    dmax = max(G.edges[e]["delta"] for e in G.edges) if G.edges else 1.0
    nodes = list(G)
    nsize = {x: 900 if x == ROOT else 250 + 900 * max(size.get(x, 0), 0) / vmax for x in nodes}
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=[C_IMG if x == ROOT else C_OTHER for x in nodes],
                           node_size=[nsize[x] for x in nodes])
    two_way = [e for e in G.edges if G.has_edge(e[1], e[0])]
    one_way = [e for e in G.edges if e not in two_way]
    for elist, cs in ((one_way, "arc3"), (two_way, "arc3,rad=0.25")):
        if not elist:
            continue
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=elist, edge_color=C_EDGE, arrows=True, arrowsize=12,
                               width=[1.0 + 3.5 * G.edges[e]["delta"] / dmax for e in elist],
                               nodelist=nodes, node_size=[nsize[x] for x in nodes], connectionstyle=cs)
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={e: G.edges[e]["label"] for e in elist},
                                     font_size=FS["edge"], connectionstyle=cs,
                                     bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    labels = {x: x if x == ROOT else textwrap.shorten(x, 22, placeholder="...") + (f" ×{mult[x]}" if mult.get(x, 1) > 1 else "")
              for x in nodes}
    for x, labs in loops.items():
        labels[x] += "\n(" + ", ".join(labs) + f" {textwrap.shorten(x, 14, placeholder='...')})"
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=FS["node"])
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
    ax.set_ylim(min(ys) - 0.22, max(ys) + 0.22)
    ax.set_title(f"Giải thích S (K* = {k_star}); số trong ngoặc: Δ điểm đồ thị ×10$^{{{expo}}}$", fontsize=FS["title"])


def case_figure(image, image_title, items, pos, suptitle, height=FIG_H):
    """Như vg_xai_plot.case_figure_items với một ảnh: hình 8.0 × 3.6 inch, ảnh : đồ thị = 1 : 1.6."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W / 2.6 * 2.6, height), gridspec_kw={"width_ratios": [1.0, 1.6]})
    _show_image(axes[0], image, image_title)
    draw_graph_pos(axes[1], items, 12, pos)
    fig.suptitle(suptitle, fontsize=FS["sup"])
    fig.tight_layout()
    return fig


def ring(names_angles, rx=1.0, ry=0.85):
    """Vật nối với ẢNH xếp trên một elip quanh ẢNH, theo góc (độ) đặt tay."""
    return {n: (rx * math.cos(math.radians(a)), ry * math.sin(math.radians(a))) for n, a in names_angles}


# ---------------------------------------------------------------------------- vị trí nút đặt tay cho từng hình
POS = {
    # Yellow flowers in a purple and yellow rain boot: hai loại ủng hai bên, cặp rain boots-flowers trải rộng ở hàng dưới
    # (nhãn hai quan hệ dài), cặp wall-stripe ở hàng trên
    "551191": {ROOT: (0.0, 0.0), "boot": (1.0, 0.05), "rain boot": (-1.0, 0.05), "rain boots": (-1.05, -0.95),
               "flowers": (1.05, -0.95), "wall": (-0.6, 0.95), "stripe": (0.6, 0.95)},
    # biohazard sign ...: bathroom, shower, door hai bên; basket phía trên với đồ trong giỏ; toilet phía dưới với đồ quanh toilet
    "2418127": {ROOT: (0.0, 0.0), "bathroom": (-1.25, 0.25), "wallpaper": (-1.55, 1.1), "shower": (1.1, 0.35),
                "door": (1.1, -0.45), "basket": (0.1, 0.95), "toilet paper": (-0.6, 1.45), "can": (0.45, 1.6),
                "counter": (1.6, 1.05), "toilet": (0.0, -0.95), "picture": (-1.25, -1.35), "mirror": (-1.75, -0.55),
                "bolt": (0.95, -1.35)},
    # brown suitcase in airport: sáu vật nối ẢNH trên elip; cặp man-vest-scarf ở hàng trên; hai cặp còn lại là cạnh dọc
    # ở hai mép dưới
    "2138299": {ROOT: (0.0, 0.0),
                **ring([("suitcase", 0), ("green suitcase", 55), ("suit case", 125), ("luggage", 180),
                        ("airport", 235), ("pillar", 305)], rx=1.4, ry=1.0),
                "vest": (-1.3, 1.7), "man": (0.0, 1.7), "scarf": (1.3, 1.7),
                "woman": (-1.45, -0.45), "blue jeans": (-1.45, -1.8), "number 2": (1.45, -0.45), "building": (1.45, -1.8)},
    # a man holding a umbrella: tám vật nối ẢNH; hair, shirt bên trái cạnh man; ground bên phải dưới rain
    "4608567": {ROOT: (0.0, 0.0), "rain": (1.15, 0.0), "umbrella": (0.8, 0.75), "fence": (0.0, 1.05),
                "pants": (-0.8, 0.75), "man": (-1.15, 0.0), "coat": (-0.55, -0.95), "wall": (0.15, -1.05),
                "image": (0.85, -0.75), "hair": (-1.75, 0.85), "shirt": (-1.8, -0.95), "ground": (1.85, -1.05)},
    # large black train (ảnh đúng): sáu vật nối ẢNH; windows chéo dưới people; cặp traffic sign-railroad track ở hàng trên
    "1317304": {ROOT: (0.0, 0.0), "people": (1.3, 0.0), "windows": (2.1, -1.3), "steam": (0.55, 0.95),
                "locomotive": (-0.55, 0.95), "train engine": (-1.55, 0.0), "light": (-0.55, -0.95),
                "train trim": (0.4, -1.05), "traffic sign": (-0.9, 1.5), "railroad track": (0.9, 1.5)},
    # A man on the road (ảnh đúng): sáu vật nối ẢNH, dog cạnh vest
    "3658532": {ROOT: (0.0, 0.0),
                **ring([("road", 0), ("man", 60), ("roadside", 120), ("farmlands", 180), ("hills", 240),
                        ("vest", 300)]),
                "dog": (1.35, -1.1)},
    # a slice of spinach and cheese pizza (ảnh đúng): bốn vật nối ẢNH, white plate cạnh crumbs, cặp greenleaf-crust phía trên
    "3650395": {ROOT: (0.0, 0.0), "spinach": (1.15, 0.35), "white cheese": (-1.45, 0.45), "crumbs": (0.6, -0.7),
                "green veggie": (-0.85, -0.75), "white plate": (1.75, -1.5), "greenleaf": (-0.6, 1.05),
                "crust": (0.65, 1.05)},
    # pink beak on bird (ảnh đúng): tám vật nối ẢNH cách đều 45 độ; eye bên phải cạnh bird head; ba cặp quan hệ không nối
    # ẢNH ở hàng trên và hàng dưới
    "2000339": {ROOT: (0.0, 0.0),
                **ring([("bird", 0), ("bird head", 45), ("beak", 90), ("feathers", 135), ("food", 180),
                        ("neck", 225), ("day", 270), ("light reflection", 315)], rx=1.3, ry=1.05),
                "eye": (2.05, 1.3), "leaf": (-1.7, 1.75), "stick": (-0.55, 1.75), "red leaf": (0.1, 1.75),
                "tree branch": (1.45, 1.75), "sticks": (-0.6, -1.75), "ground": (0.6, -1.75)},
    # woman wearing green shorts (ảnh đúng): woman bên trái với bốn quan hệ; man chéo dưới knee brace bên phải
    "1939039": {ROOT: (0.35, 0.0), "shorts": (0.35, -1.05), "woman": (-0.85, 0.0), "hair": (-0.2, 1.0),
                "shoes": (-1.65, -0.8), "arms": (-1.8, 0.95), "knee brace": (1.35, -0.75), "man": (2.15, -1.6),
                "shirt": (0.45, 1.1), "tree": (1.55, 0.2)},
}

# Mọi hình phụ lục dùng khung cao 4.8 inch như Hình 6 (ca Huế) ở thân bài, để đồ thị nhiều nút không chồng nhãn.
DENSE_H = 4.8


def report_overlaps(fig, ax, name):
    """In các cặp chữ chồng nhau trong khung đồ thị (để chỉnh tay vị trí nút)."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in texts]
    bad = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            x = Bbox.intersection(boxes[i], boxes[j])
            if x is not None and x.width * x.height > 20:
                bad.append((texts[i].get_text().replace("\n", " ")[:24], texts[j].get_text().replace("\n", " ")[:24],
                            int(x.width * x.height)))
    print(f"  {name}: {len(bad)} cặp chồng" + ("".join(f"\n      {a!r} x {b!r} ({s})" for a, b, s in bad)))

LOAI = dict(thanh_cong="thanh_cong", that_bai_lam_hai="that_bai_lam_hai", that_bai_khong_cuu="that_bai_khong_cuu")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for c in read_vg(INP / "V6_cases.md"):
        if c["id"] == "575905":          # ca peace sign đã là Hình 2 ở thân bài
            continue
        g = c["gold"]
        img = Image.open(INP / f"vg_{c['id']}_{'gold' if 'top' in c else 'top'}.jpg")
        sup = f"[{LOAI[c['loai']]}] {c['query']}\nCLIP hạng {c['clip']} → CLIP + GAT hạng {c['model']}"
        fig = case_figure(img, f"Ảnh đúng, hạng {g['rank']}\nbỏ giải thích S: hạng {g['rank_rm']}", vg_items(g["S"]),
                          POS[c["id"]], sup, height=DENSE_H)
        report_overlaps(fig, fig.axes[1], c["id"])
        fig.savefig(OUT / f"vg_{c['id']}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("đã vẽ", c["id"], c["query"])
