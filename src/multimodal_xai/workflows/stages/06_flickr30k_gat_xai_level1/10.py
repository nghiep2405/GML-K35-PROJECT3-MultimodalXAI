rank_table = pd.DataFrame({
    "query_index": np.arange(len(test["caption_ids"])),
    "caption_id": test["caption_ids"],
    "positive_image_id": test["positive_image_ids"],
    "caption": test["captions"],
    "rank_clip": rank_of_target(CLIP_TEST, T_TARGET).numpy(),
    "top1_clip": [test["image_ids"][i] for i in CLIP_TEST.argmax(1).tolist()],
    "clip_score_pos": CLIP_TEST.gather(1, T_TARGET[:, None]).squeeze(1).numpy(),
})
for name in MODELS:
    rank_table[f"rank_{name}"] = rank_of_target(FUSED_TEST[name], T_TARGET).numpy()
    rank_table[f"top1_{name}"] = [test["image_ids"][i] for i in FUSED_TEST[name].argmax(1).tolist()]
    rank_table[f"graph_score_pos_{name}"] = GRAPH_TEST[name].gather(1, T_TARGET[:, None]).squeeze(1).numpy()
rank_table.to_csv(OUT / "B3_test_query_ranks.csv", index=False)

def _share(mask):
    return f"{int(mask.sum())} ({mask.mean():.1%})"
r = rank_table
B3_counts = pd.Series({
    "CLIP đúng top-1": _share(r.rank_clip == 1),
    f"{MAIN_RUN} đúng top-1": _share(r[f"rank_{MAIN_RUN}"] == 1),
    "CLIP sai → GAT đúng (đồ thị cứu)": _share((r.rank_clip > 1) & (r[f"rank_{MAIN_RUN}"] == 1)),
    "  trong đó no_edge cũng đúng": _share((r.rank_clip > 1) & (r[f"rank_{MAIN_RUN}"] == 1) & (r.rank_no_edge == 1)),
    "  trong đó cần quan hệ thật (no_edge & random sai)": _share(
        (r.rank_clip > 1) & (r[f"rank_{MAIN_RUN}"] == 1) & (r.rank_no_edge > 1) & (r.rank_random_edges > 1)),
    "CLIP đúng → GAT sai (đồ thị làm hại)": _share((r.rank_clip == 1) & (r[f"rank_{MAIN_RUN}"] > 1)),
    "Cả hai cùng sai": _share((r.rank_clip > 1) & (r[f"rank_{MAIN_RUN}"] > 1)),
}, name="số truy vấn / 5000")
B3_counts.to_csv(OUT / "B3_counts.csv")
B3_counts
