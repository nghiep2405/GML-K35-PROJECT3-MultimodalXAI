def build_graphs(split, edge_mode, seed):
    return [
        gat.build_graph(nodes, metadata, index, edge_mode=edge_mode, seed=seed)
        for index, (nodes, metadata) in enumerate(zip(split["graph_nodes"], split["graph_metadata"]))
    ]


def rank_of_target(scores, targets):
    positive = scores.gather(1, targets[:, None])
    return 1 + (scores > positive).sum(dim=1)

TEST_GRAPHS = {}
for name, (_, edge_mode, seed) in CKPT_DIRS.items():
    key = (edge_mode, seed if edge_mode == "random_edges" else 0)
    if key not in TEST_GRAPHS:
        TEST_GRAPHS[key] = build_graphs(test, edge_mode, seed)
def test_graphs_for(name):
    _, edge_mode, seed = CKPT_DIRS[name]
    return TEST_GRAPHS[(edge_mode, seed if edge_mode == "random_edges" else 0)]

T_TEXT = test["text_embeddings"].float()
T_IMG = test["global_image_embeddings"].float()
T_TARGET = test["targets"]
CLIP_TEST = T_TEXT @ T_IMG.T                                # [5000, 1000]
GRAPH_TEST, FUSED_TEST, QPROJ_TEST, GEMB_TEST = {}, {}, {}, {}
for name, model in MODELS.items():
    g_img = gat.encode_candidates(model, test_graphs_for(name)).float()
    q_proj = model.encode_query(T_TEXT.to(DEVICE)).float().cpu()
    GEMB_TEST[name], QPROJ_TEST[name] = g_img, q_proj
    GRAPH_TEST[name] = q_proj @ g_img.T
    FUSED_TEST[name] = (1 - ALPHA[name]) * CLIP_TEST + ALPHA[name] * GRAPH_TEST[name]

summary_path = FINAL / "results/Flickr30k/final_test_summary.json"
reference = {r["run"]: r for r in json.loads(summary_path.read_text())["runs"]}
rows, TOL = [], 2e-4
for name in MODELS:
    ours = gat.ranking_metrics(FUSED_TEST[name], T_TARGET)
    ref = reference[name]["test_clip_plus_graph"]
    rows.append({"run": name, "R@1 (tái lập)": ours["Recall@1"], "R@1 (nhóm)": ref["Recall@1"],
                 "MRR (tái lập)": ours["MRR"], "MRR (nhóm)": ref["MRR"],
                 "lệch tối đa": max(abs(ours["Recall@1"] - ref["Recall@1"]), abs(ours["MRR"] - ref["MRR"]))})
clip_m = gat.ranking_metrics(CLIP_TEST, T_TARGET)
rows.append({"run": "CLIP", "R@1 (tái lập)": clip_m["Recall@1"],
             "R@1 (nhóm)": reference[MAIN_RUN]["test_clip_only"]["Recall@1"], "MRR (tái lập)": clip_m["MRR"],
             "MRR (nhóm)": reference[MAIN_RUN]["test_clip_only"]["MRR"],
             "lệch tối đa": max(abs(clip_m["Recall@1"] - reference[MAIN_RUN]["test_clip_only"]["Recall@1"]),
                                abs(clip_m["MRR"] - reference[MAIN_RUN]["test_clip_only"]["MRR"]))})
B2 = pd.DataFrame(rows)
B2.to_csv(OUT / "B2_reproduction_check.csv", index=False)
assert (B2["lệch tối đa"] <= TOL).all(), "Không tái lập được số test — DỪNG và kiểm tra lại dữ liệu/checkpoint."
print("B2: tái lập khớp với final_test_summary.json (ngưỡng", TOL, ")")
B2
