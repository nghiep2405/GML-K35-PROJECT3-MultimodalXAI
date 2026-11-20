# Hình giải thích cấp 1, cùng kiểu với hình ca cấp 2 (07 / 18.py):
# ảnh | đồ thị giải thích (spring layout, nét xanh dương = quan hệ trong S, độ dày và số trong ngoặc = Δ điểm đồ thị).
# Chỉ vẽ đúng tập giải thích S. Nút trùng tên được gộp (×n); quan hệ trùng (đầu, quan hệ, đuôi) cộng Δ.
import math
import textwrap
from collections import OrderedDict

import matplotlib.pyplot as plt
import networkx as nx

# Cùng bảng màu với các biểu đồ của báo cáo (report/figures/scripts/make_figures.py):
# xanh dương = tập giải thích S, cam = ảnh/thực thể được trả về, xanh ngọc = thực thể neo, xám = nút khác.
C_EDGE, C_IMG, C_ANCHOR, C_OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#d9d8d4"
C_LINK, C_QUERY = "#9cc0ef", "#52514e"
ROOT = "ẢNH"
# Cỡ chữ (pt) cho hình rộng FIG_W inch; khi chèn bằng \textwidth (7.16 in) chữ còn khoảng 6 đến 8 pt.
FS = dict(sup=11, title=10, node=9, edge=8.5, tick=9, label=9)
FIG_W, FIG_H = 8.0, 3.6


def _endpoints(g, u):
    """Chiều gốc của quan hệ (đầu, đuôi) theo chú thích Visual Genome."""
    e = g.unit_edges[u][0]
    s, d = int(g.src[e]), int(g.dst[e])
    if bool(g.rev[e]):
        s, d = d, s
    return s, d


def _rel(g, u, rel_name):
    r = rel_name.get(g.unit_keys[u][2], str(g.unit_keys[u][2]))
    return "quan hệ lạ" if r == "UNK_REL" else r


def merged_explanation(g, S, deltas, node_label, rel_name):
    """Trả về list ((đầu, quan hệ, đuôi), n, tổng Δ) sắp theo Δ giảm dần; đầu/đuôi là nhãn, ảnh gốc là ROOT."""
    key_of = lambda v: ROOT if v == g.root else node_label(g, v)
    groups = OrderedDict()
    for u in S:
        s, d = _endpoints(g, u)
        k = (key_of(s), _rel(g, u, rel_name), key_of(d))
        n, t = groups.get(k, (0, 0.0))
        groups[k] = (n + 1, t + float(deltas[u]))
    return sorted(((k, n, t) for k, (n, t) in groups.items()), key=lambda x: -x[2])


def _sentence(k):
    s, r, d = k
    return f"ảnh chứa {d}" if s == ROOT and r == "contains" else f"{s} {r} {d}"


def _layout(G, tries=30):
    """Spring layout (như cấp 2), chọn seed để không cạnh nào đi xuyên qua nút khác và các nút không quá sát nhau."""
    import numpy as np

    def penalty(pos):
        P = {k: np.asarray(v, float) for k, v in pos.items()}
        span = max(np.ptp([p[0] for p in P.values()]), np.ptp([p[1] for p in P.values()]), 1e-9)
        bad = 0.0
        for a, b in G.edges:
            A, B = P[a], P[b]
            AB = B - A
            for c, C in P.items():
                if c in (a, b):
                    continue
                t = np.clip(np.dot(C - A, AB) / max(np.dot(AB, AB), 1e-12), 0, 1)
                dist = np.linalg.norm(A + t * AB - C) / span
                bad += max(0.0, 0.12 - dist) * 10
        keys = list(P)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                bad += max(0.0, 0.15 - np.linalg.norm(P[keys[i]] - P[keys[j]]) / span) * 5
        return bad

    best, best_pen = None, float("inf")
    for seed in range(tries):
        pos = nx.spring_layout(G, seed=seed, k=1.2)
        pen = penalty(pos)
        if pen < best_pen:
            best, best_pen = pos, pen
        if pen == 0:
            break
    return best


def draw_graph(ax, items, k_star):
    ax.axis("off")
    if not items:
        ax.text(.5, .5, "Không có quan hệ nào\nlàm tăng điểm đồ thị", ha="center", va="center", fontsize=FS["title"],
                transform=ax.transAxes)
        return
    G = nx.DiGraph()
    G.add_node(ROOT)
    size, mult, loops = {}, {}, {}
    # Δ ghi trên cạnh theo một thang 10^expo chung; độ dày cạnh tỉ lệ với Δ.
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
        if s == d:                # hai vật cùng tên có quan hệ với nhau: ghi ngay dưới tên nút
            loops.setdefault(s, []).append(lab)
            continue
        if G.has_edge(s, d):      # nhiều quan hệ trên cùng một cặp: ghi đủ, không để nhãn sau đè nhãn trước
            G.edges[s, d]["label"] += " / " + lab
            G.edges[s, d]["delta"] += t
        else:
            G.add_edge(s, d, label=lab, delta=t)
    for e in G.edges:
        G.edges[e]["label"] = textwrap.shorten(G.edges[e]["label"], 40, placeholder="...")
    vmax = max(size.values()) if size else 1.0
    dmax = max(G.edges[e]["delta"] for e in G.edges) if G.edges else 1.0
    pos = _layout(G)
    nodes = list(G)
    nsize = {x: 900 if x == ROOT else 250 + 900 * max(size.get(x, 0), 0) / vmax for x in nodes}
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=[C_IMG if x == ROOT else C_OTHER for x in nodes],
                           node_size=[nsize[x] for x in nodes])
    # cặp có quan hệ theo cả hai chiều: uốn cong để hai cạnh và hai nhãn tách nhau
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
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45); ax.set_ylim(min(ys) - 0.22, max(ys) + 0.22)
    ax.set_title(f"Giải thích S (K* = {k_star}); số trong ngoặc: Δ điểm đồ thị ×10$^{{{expo}}}$",
                 fontsize=FS["title"])


def _show_image(ax, image, title):
    if image is not None:
        ax.imshow(image)
    else:
        ax.text(.5, .5, "không tải được ảnh", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=FS["title"])
    ax.axis("off")


def case_figure_items(images, items, k_star, suptitle):
    """Như case_figure nhưng nhận thẳng items của merged_explanation (dùng khi vẽ lại từ V6_cases.md)."""
    n = len(images)
    fig, axes = plt.subplots(1, n + 1, figsize=(FIG_W / 2.6 * (n + 1.6), FIG_H),
                             gridspec_kw={"width_ratios": [1.0] * n + [1.6]})
    for ax, (title, img) in zip(axes, images):
        _show_image(ax, img, title)
    draw_graph(axes[n], items, k_star)
    fig.suptitle(suptitle, fontsize=FS["sup"])
    fig.tight_layout()
    return fig


def case_figure(images, g, S, deltas, node_label, rel_name, k_star, suptitle):
    """images: list (tiêu đề, ảnh PIL hoặc None); giải thích (g, S, deltas) là của ảnh đầu tiên, như cấp 2."""
    return case_figure_items(images, merged_explanation(g, S, deltas, node_label, rel_name), k_star, suptitle)


def plot_explained_row(axes, g, S, deltas, image, header, node_label, rel_name, k_star):
    """Một hàng (ảnh, đồ thị) cho demo nhiều kết quả."""
    _show_image(axes[0], image, header)
    items = merged_explanation(g, S, deltas, node_label, rel_name)
    draw_graph(axes[1], items, k_star)


def new_figure(n_rows):
    fig, axes = plt.subplots(n_rows, 2, figsize=(FIG_W, FIG_H * n_rows), squeeze=False,
                             gridspec_kw={"width_ratios": [1.0, 1.6]})
    return fig, axes


def unit_sentence(g, u, node_label, rel_name):
    s, d = _endpoints(g, u)
    r = _rel(g, u, rel_name)
    if r == "contains":
        return f"ảnh chứa {node_label(g, d if s == g.root else s)}"
    return f"{node_label(g, s)} {r} {node_label(g, d)}"


def vg_image_urls(image_id):
    """Đường dẫn ảnh Visual Genome công khai theo số hiệu ảnh (hai thư mục của bộ dữ liệu gốc)."""
    num = str(image_id).split(":")[-1]
    return [f"https://cs.stanford.edu/people/rak248/VG_100K/{num}.jpg",
            f"https://cs.stanford.edu/people/rak248/VG_100K_2/{num}.jpg"]
