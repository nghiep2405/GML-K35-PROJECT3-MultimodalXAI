import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

IMG_INDEX = {iid: i for i, iid in enumerate(test["image_ids"])}

def _open_image(image_id):
    if not IMAGE_DIR:
        return None
    p = Path(IMAGE_DIR) / f"{image_id}.jpg"
    if not p.exists():
        return None
    from PIL import Image
    return Image.open(p).convert("RGB")

def draw_layout(ax, split, image_index, exp, labels, title):
    meta = split["graph_metadata"][image_index]
    W, H = meta["width"], meta["height"]
    img = _open_image(split["image_ids"][image_index])
    if img is not None:
        ax.imshow(img)
    else:
        ax.add_patch(patches.Rectangle((0, 0), W, H, color="#F1F5F9"))
        ax.text(W / 2, H * 0.02, "(không có ảnh gốc — chỉ vẽ bố cục hộp)", ha="center", va="top", fontsize=7,
                color="#64748B")
    node_score = {}
    for rel in exp["relations"]:
        if rel["in_explanation"]:
            for n in rel["nodes"]:
                if n:
                    node_score[n] = node_score.get(n, 0) + rel["delta_graph"]
    vmax = max(node_score.values()) if node_score else 1.0
    for r_i, region in enumerate(meta["regions"]):
        x1, y1, x2, y2 = region["box_pixels"]
        n = r_i + 1
        on = n in node_score
        color = plt.cm.Reds(0.35 + 0.65 * node_score[n] / vmax) if on else "#94A3B8"
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2.2 if on else 0.8,
                                       ec=color, ls="-" if on else "--"))
        if on:
            ax.text(x1 + 2, y1 + 2, labels.get(n, f"R{r_i}").split(":")[0], fontsize=7, color="white",
                    va="top", bbox=dict(fc=color, ec="none", pad=1))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9)

def draw_subgraph(ax, exp, labels, title):
    G = nx.DiGraph()
    G.add_node("ẢNH")
    edge_lab = {}
    for rel in exp["relations"]:
        if not rel["in_explanation"]:
            continue
        u, v = rel["nodes"]
        nu = "ẢNH" if u == 0 else labels.get(u, f"R{u - 1}")
        nv = labels.get(v, f"R{v - 1}")
        if rel["kind"] == "star":
            G.add_edge(nv, "ẢNH", w=rel["delta_graph"])
            edge_lab[(nv, "ẢNH")] = f"thuộc\nΔ={rel['delta_graph']:+.3f}"
        else:
            t = rel["type_ids"].get(f"{u}->{v}")
            G.add_edge(nu, nv, w=rel["delta_graph"])
            edge_lab[(nu, nv)] = f"{REL_SHORT.get(t, '?')}\nΔ={rel['delta_graph']:+.3f}"
            for n in (nu, nv):
                if not G.has_edge(n, "ẢNH"):
                    G.add_edge(n, "ẢNH", w=0.0)
    outer = [n for n in G if n != "ẢNH"]
    pos = {"ẢNH": (0.0, 0.0)}
    for i, n in enumerate(outer):   # không đặt 2 nút đối xứng qua tâm, để cạnh vùng–vùng không bị nút ẢNH che
        ang = math.pi / 2 + 2 * math.pi * i / (len(outer) + 0.5)
        pos[n] = (math.cos(ang), math.sin(ang))
    main = [(u, v) for u, v in G.edges if (u, v) in edge_lab]
    helper = [(u, v) for u, v in G.edges if (u, v) not in edge_lab]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=["#F59E0B" if n == "ẢNH" else "#BFDBFE" for n in G],
                           node_size=900, edgecolors="#1E3A5F")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n.replace(":", "\n", 2) for n in G}, font_size=6)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=helper, width=0.7, edge_color="#CBD5E1", style="dashed",
                           arrows=True, arrowsize=8, node_size=900)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=main, width=[max(0.8, 40 * G.edges[e]["w"]) for e in main],
                           edge_color=["#DC2626" if G.edges[e]["w"] > 0 else "#94A3B8" for e in main],
                           arrowsize=11, node_size=900)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_lab, font_size=6, font_color="#7F1D1D",
                                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.text(0.0, -0.02, "đỏ = quan hệ trong giải thích (độ dày ∝ Δ) · nét đứt = đường đi tiếp về ẢNH qua cạnh vùng–ảnh",
            transform=ax.transAxes, fontsize=6, color="#475569")
    ax.set_title(title, fontsize=9); ax.axis("off")

def explain_test(name, qi, image_index, K):
    # Chỉ ảnh đúng mới chứa câu truy vấn trong 5 caption của nó → chỉ loại câu đó khi gắn tên vùng cho ảnh đúng.
    own = image_index == int(T_TARGET[qi])
    labels = region_labels(test, image_index, query_sentence_index(test["caption_ids"][qi]) if own else None)
    ex = explain(MODELS[name], test_graphs_for(name)[image_index], QPROJ_TEST[name][qi], ALPHA[name], K, labels)
    ex["labels"] = labels
    return ex

def fused_rank_after(name, qi, image_index, new_graph_score):
    a = ALPHA[name]
    fused = FUSED_TEST[name][qi].clone()
    fused[image_index] = (1 - a) * CLIP_TEST[qi, image_index] + a * new_graph_score
    return int(1 + (fused > fused[image_index]).sum())

def auto_observations(row, ex_pos, ex_other, other_role):
    obs = []
    meta = test["graph_metadata"][int(T_TARGET[row.query_index])]
    struct, sg = ex_pos["structure_contribution"], ex_pos["S_full"]
    n_sp = ex_pos["n_spatial_in_explanation"]
    obs.append(f"Giải thích gồm {ex_pos['n_explanation']} quan hệ, trong đó {n_sp} quan hệ không gian giữa các vùng "
               f"và {ex_pos['n_explanation'] - n_sp} quan hệ vùng–ảnh.")
    obs.append(f"Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: {struct:+.4f} "
               f"(điểm đồ thị {sg:.4f}; sau trộn α={ALPHA[MAIN_RUN]} thành {ALPHA[MAIN_RUN] * struct:+.4f} điểm tổng).")
    if ex_pos["n_explanation"] == 0:
        obs.append("Không có quan hệ nào đóng góp dương: điểm đồ thị của ảnh đúng đến từ chính các nút, không từ quan hệ.")
    if len(meta["regions"]) <= 2:
        obs.append(f"Ảnh đúng chỉ có {len(meta['regions'])} vùng → đồ thị nghèo quan hệ.")
    big = [r_i for r_i, rg in enumerate(meta["regions"])
           if (rg["box_normalized"][2] - rg["box_normalized"][0]) * (rg["box_normalized"][3] - rg["box_normalized"][1]) > 0.8]
    if big:
        obs.append(f"Có {len(big)} vùng phủ >80% ảnh (R{', R'.join(map(str, big))}) → quan hệ 'chồng lấn' với mọi vùng, ít thông tin.")
    if ex_other is not None:
        diff = ex_other["S_full"] - ex_pos["S_full"]
        obs.append(f"Điểm đồ thị của {other_role} {'cao' if diff > 0 else 'thấp'} hơn ảnh đúng {abs(diff):.4f}; "
                   f"đóng góp cấu trúc của nó {ex_other['structure_contribution']:+.4f}.")
    return obs

# Nhận xét nguyên nhân cho 10 ca (viết từ lần chạy tham chiếu, dựa trên số đo và annotation Flickr30k Entities).
# Gắn theo caption_id: nếu lần chạy của bạn chọn ra ca khác (rất hiếm, do sai số GPU), ca đó sẽ hiện "chưa có nhận xét".
TEAM_NOTES = {'6160193920#3': 'Đồ thị kéo ảnh đúng từ hạng 127 lên 1. Đóng góp lớn nhất là vùng áo xanh ("a blue top") được '
                 'truyền vào nút ảnh (Δ = +0.045), khớp với cụm "blue print shirt" của truy vấn. Quan hệ không '
                 'gian "áo nằm phía dưới nụ cười" chỉ đóng góp +0.001. Ảnh CLIP từng xếp đầu cũng có vùng áo xanh '
                 'nhưng điểm đồ thị thấp hơn 0.263, nên bị đẩy xuống. Thành công cần quan hệ thật (no_edge hạng '
                 '4, random_edges hạng 13), nhưng phần lớn đến từ cạnh vùng–ảnh. Bỏ 2 quan hệ trong giải thích '
                 'thì ảnh đúng tụt xuống hạng 2.',
 '4980795412#1': 'Ca có đóng góp cấu trúc lớn nhất (+0.384): bỏ mọi quan hệ thì điểm đồ thị gần như về 0. Các '
                 'quan hệ dương khớp đúng ngữ nghĩa truy vấn: vùng "a green shirt" (+0.031), "a man" (+0.016) và '
                 '"người chồng lấn áo xanh" (+0.005), tức "người mặc áo xanh". Các vùng nền (tòa nhà, lề đường) '
                 'đóng góp âm. Ảnh CLIP xếp đầu có điểm đồ thị âm (−0.077), nên CLIP + GAT đổi thứ tự. Bỏ 8 quan '
                 'hệ giải thích thì ảnh đúng tụt xuống hạng 3.',
 '3815318559#1': 'Đồ thị nâng ảnh từ hạng 33 lên 1 nhờ các vùng người, cửa sổ, đường phố và quan hệ chồng lấn '
                 'người–mũ–cửa sổ (tương ứng "person looking out of"). Vùng "a red car" không nằm trong giải '
                 'thích, quan hệ "xe đỏ chồng lấn mũ" còn đóng góp âm: mô hình chưa khai thác thuộc tính màu '
                 '"red" của truy vấn. Đối thủ (ảnh nhiều xe đỗ, tòa nhà) có đóng góp cấu trúc âm (−0.095), nên '
                 'một phần thành công đến từ việc đồ thị phạt đối thủ. Bỏ giải thích → hạng 7.',
 '4821810079#2': 'Ảnh chỉ có 3 vùng (6 quan hệ). Vùng "a red shirt" đóng góp chính (+0.045), cộng quan hệ "người '
                 'chồng lấn áo đỏ" (+0.003), khớp "a man in a red shirt". Vùng "a blue market stall" và quan hệ '
                 'của nó với áo đỏ đóng góp âm. Quan hệ hành động "holding" không có trong đồ thị không gian, nên '
                 'phần "holding a red piece of clothing" không được giải thích. Bỏ 2 quan hệ giải thích → hạng 5. '
                 'no_edge chỉ đạt hạng 3.',
 '6225843702#1': 'Giải thích chỉ gồm 4 cạnh vùng–ảnh (người đạp xe, xe đạp, chướng ngại vật), không có quan hệ '
                 'không gian nào. Bỏ cả 4 quan hệ thì ảnh vẫn đứng hạng 1, vì đóng góp cấu trúc nhỏ (+0.077) so '
                 'với điểm từ chính các nút (0.375). Ca này cho thấy giới hạn: thành công không phải lúc nào cũng '
                 'do quan hệ giữa các vùng, mà chủ yếu do biểu diễn vùng được gộp vào nút ảnh. Đối thủ (có biển '
                 'báo, xe đạp nằm) có đóng góp cấu trúc âm.',
 '2319389144#1': 'Đồ thị làm hại: CLIP hạng 1 → CLIP + GAT hạng 40. Điểm đồ thị của ảnh đúng âm (−0.093) và âm '
                 'ngay cả khi bỏ mọi quan hệ (−0.080), nên lỗi nằm ở biểu diễn nút của ảnh nhìn từ trên cao với '
                 'các vùng người rất nhỏ. Ảnh bị xếp nhầm lên đầu có 104 quan hệ, nhiều vùng "cars" và "colorful '
                 'buildings", điểm đồ thị +0.296. Đồ thị thiên vị ảnh đường phố nhiều vùng hơn là đúng cảnh '
                 '"looks down on a street".',
 '4686568156#4': 'Truy vấn trừu tượng ("two lovers just met in a date"): không vùng hay quan hệ không gian nào '
                 'biểu diễn được khái niệm "hẹn hò". Vùng "a woman\'s face" đóng góp âm (−0.010) và kéo theo quan '
                 'hệ chồng lấn áo khoác–mặt (−0.006). Điểm đồ thị ảnh đúng âm (−0.057), trong khi ảnh nhầm có cặp '
                 '"man chồng lấn woman" và trang phục đỏ. Đây là loại truy vấn gián tiếp mà đồ thị không gian '
                 'không hỗ trợ được, cần quan hệ ngữ nghĩa (cấp độ 2–3).',
 '2735558076#3': 'CLIP hạng 1 → GAT hạng 20. Vùng "pavement" (khớp "sidewalk") lại đóng góp âm (−0.005), vùng '
                 '"black boots" âm (−0.007). Ảnh đúng có 44 quan hệ nhưng đóng góp cấu trúc âm (−0.019). Ảnh nhầm '
                 '(một cặp đôi, áo, quần short) có đóng góp cấu trúc +0.114 nhờ các vùng trang phục rõ ràng. Mô '
                 'hình ưu tiên vùng quần áo hơn bối cảnh "đang chờ trên vỉa hè".',
 '2877088081#2': 'Suýt đúng: đồ thị nâng ảnh từ hạng 82 lên 2 (điểm đồ thị 0.551, cao hơn ảnh xếp đầu 0.196) '
                 'nhưng CLIP của ảnh xếp đầu (trượt ván, kính râm, mặt đường) cao hơn nên với α = 0.2 vẫn thắng. '
                 'Vùng đóng góp chính là vùng mũ bảo hiểm (theo annotation, chỉ được nhắc trong chính câu truy '
                 'vấn nên hình chỉ ghi loại "clothing*"); vùng "A boy" đóng góp âm. no_edge xếp đúng hạng 1: ở ca '
                 'này quan hệ thật không giúp thêm.',
 '1775223211#2': 'Suýt đúng: hạng CLIP 59 → 2 (random_edges lại xếp đúng hạng 1). Đóng góp chính đến từ vùng "the '
                 'thumbs up" (theo annotation, chỉ có trong câu truy vấn nên hình ghi "other*") và vùng "cloth"; '
                 'các vùng "the unfinished item", "a loom" đóng góp âm. Ảnh nhầm cũng có người phụ nữ, vùng "her '
                 'hand" và vùng "something" với đóng góp cấu trúc lớn (+0.184). Nội dung đặc thù của truy vấn '
                 '("thumbs up", bối cảnh y tế) không có quan hệ tương ứng trong đồ thị không gian.'}

ANALYSIS_MD = ["# B9 — Phân tích chi tiết 5 truy vấn thành công và 5 truy vấn thất bại\n",
               f"Mô hình: `{MAIN_RUN}`, α_graph = {ALPHA[MAIN_RUN]}, K = {K_STAR} (chọn trên validation).\n"]
CASE_RESULTS = []
for row in CASES.itertuples():
    qi = int(row.query_index)
    pos_idx = int(T_TARGET[qi])
    ex_pos = explain_test(MAIN_RUN, qi, pos_idx, K_STAR)
    if row.case == "failure":
        other_idx, role = IMG_INDEX[getattr(row, f"top1_{MAIN_RUN}")], "ảnh bị xếp nhầm lên đầu"
    elif row.case == "success":
        other_idx, role = IMG_INDEX[row.top1_clip], "ảnh CLIP từng xếp đầu"
    else:
        top1 = IMG_INDEX[getattr(row, f"top1_{MAIN_RUN}")]
        other_idx, role = (top1, "ảnh xếp đầu") if top1 != pos_idx else (None, None)
    ex_other = explain_test(MAIN_RUN, qi, other_idx, K_STAR) if other_idx is not None else None

    rank_del = fused_rank_after(MAIN_RUN, qi, pos_idx, ex_pos["S_delete_explanation"])
    nrow = 2 if ex_other is not None else 1
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 5.2 * nrow), dpi=130, squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    draw_layout(axes[0, 0], test, pos_idx, ex_pos, ex_pos["labels"],
                f"Ảnh đúng {test['image_ids'][pos_idx]} | hạng CLIP {row.rank_clip} → CLIP+GAT {getattr(row, R_MAIN)}")
    draw_subgraph(axes[0, 1], ex_pos, ex_pos["labels"],
                  f"Giải thích (K={K_STAR}) | S_graph={ex_pos['S_full']:.3f}, cấu trúc={ex_pos['structure_contribution']:+.3f}"
                  f" | bỏ giải thích → hạng {rank_del}")
    if ex_other is not None:
        draw_layout(axes[1, 0], test, other_idx, ex_other, ex_other["labels"],
                    f"{role.capitalize()} {test['image_ids'][other_idx]}")
        draw_subgraph(axes[1, 1], ex_other, ex_other["labels"],
                      f"Giải thích | S_graph={ex_other['S_full']:.3f}, cấu trúc={ex_other['structure_contribution']:+.3f}")
    fig.suptitle(f"[{row.case_id}] \"{row.caption}\"", fontsize=10, y=1.0)
    fig.tight_layout()
    fig_path = OUT / "cases" / f"{row.case_id}_{row.caption_id.replace('#', '_')}.png"
    fig.savefig(fig_path, bbox_inches="tight"); plt.close(fig)

    obs = auto_observations(row, ex_pos, ex_other, role)
    strip = lambda e: None if e is None else {k: v for k, v in e.items() if not k.startswith("_")}
    CASE_RESULTS.append({
        "case_id": row.case_id, "case": row.case, "failure_type": getattr(row, "failure_type", None),
        "query_index": qi, "caption_id": row.caption_id, "caption": row.caption,
        "positive_image_id": row.positive_image_id,
        "ranks": {"clip": int(row.rank_clip), MAIN_RUN: int(getattr(row, R_MAIN)),
                  "no_edge": int(row.rank_no_edge), "random_edges": int(row.rank_random_edges)},
        "scores_positive": {"clip": float(CLIP_TEST[qi, pos_idx]), "graph": ex_pos["S_full"],
                            "fused": float(FUSED_TEST[MAIN_RUN][qi, pos_idx])},
        "rank_after_deleting_explanation": rank_del,
        "explanation_positive": strip(ex_pos),
        "other_role": role, "other_image_id": None if other_idx is None else test["image_ids"][other_idx],
        "explanation_other": strip(ex_other), "observations": obs, "figure": fig_path.name,
    })
    if row.case in ("success", "failure"):
        top = [rel for rel in ex_pos["relations"] if rel["in_explanation"]]
        ANALYSIS_MD += [f"\n## {row.case_id}" + (f" ({row.failure_type})" if row.case == "failure" else ""),
                        f"**Truy vấn:** \"{row.caption}\"  \n**Ảnh đúng:** {row.positive_image_id}  \n"
                        f"**Hạng:** CLIP {row.rank_clip} · CLIP+GAT {getattr(row, R_MAIN)} · no_edge {row.rank_no_edge}"
                        f" · random_edges {row.rank_random_edges}  \n"
                        f"**Điểm ảnh đúng:** CLIP {CLIP_TEST[qi, pos_idx]:.4f} · Graph {ex_pos['S_full']:.4f} · "
                        f"Tổng {FUSED_TEST[MAIN_RUN][qi, pos_idx]:.4f}  \n"
                        f"**Bỏ các quan hệ trong giải thích:** điểm đồ thị {ex_pos['S_full']:.4f} → "
                        f"{ex_pos['S_delete_explanation']:.4f}, hạng {getattr(row, R_MAIN)} → {rank_del}  \n"
                        f"**Hình:** `cases/{fig_path.name}`\n",
                        "| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |", "|---|---|---|---:|---:|"]
        ANALYSIS_MD += [f"| {i + 1} | {rel['relation']} | {rel['path']} | {rel['delta_graph']:+.4f} | "
                        f"{rel['delta_fused']:+.4f} |" for i, rel in enumerate(top)] or ["| – | (không có) | | | |"]
        ts = ex_pos["top_spatial"]
        ANALYSIS_MD += ["", f"**Quan hệ không gian mạnh nhất:** {ts['relation']} (Δ = {ts['delta_graph']:+.4f})"
                        if ts else "", "**Quan sát tự động:**"] + [f"- {o}" for o in obs] + \
                       ["", "**Nhận xét (nguyên nhân):** " + TEAM_NOTES.get(row.caption_id,
                        "_…chưa có nhận xét cho truy vấn này (ca được chọn khác lần chạy tham chiếu) — điền sau khi xem hình…_"), ""]

(OUT / "B9_cases.json").write_text(json.dumps(CASE_RESULTS, indent=1, ensure_ascii=False), encoding="utf-8")
(OUT / "B9_case_analysis.md").write_text("\n".join(ANALYSIS_MD), encoding="utf-8")
print(f"Đã lưu {len(CASE_RESULTS)} ca vào {OUT / 'cases'}")
pd.DataFrame([{"case_id": c["case_id"], "hạng CLIP": c["ranks"]["clip"], "hạng GAT": c["ranks"][MAIN_RUN],
               "cấu trúc": round(c["explanation_positive"]["structure_contribution"], 4),
               "số quan hệ giải thích": c["explanation_positive"]["n_explanation"],
               "hạng sau khi bỏ giải thích": c["rank_after_deleting_explanation"]} for c in CASE_RESULTS])
