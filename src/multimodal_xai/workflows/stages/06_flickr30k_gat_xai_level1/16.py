VAL_GRAPHS = build_graphs(val, "real_edges", 0)
V_TEXT = val["text_embeddings"].float()
V_QPROJ = MODELS[MAIN_RUN].encode_query(V_TEXT.to(DEVICE)).float().cpu()
v_rng = np.random.default_rng(SELECTION_SEED)
val_q = v_rng.choice(len(val["caption_ids"]), min(N_VAL_QUERIES_B6, len(val["caption_ids"])), replace=False)

b6_rows = []
for qi in val_q:
    img = int(val["targets"][qi])
    data = VAL_GRAPHS[img]
    groups = relation_groups(data)
    if not groups:
        continue
    E = data.edge_index.shape[1]
    is_self = data.edge_index[0] == data.edge_index[1]
    masks = [torch.ones(E, dtype=torch.bool), is_self.clone()]
    for g in groups:
        m = torch.ones(E, dtype=torch.bool); m[g["edges"]] = False; masks.append(m)
    s = scores_with_masks(MODELS[MAIN_RUN], data, V_QPROJ[qi], torch.stack(masks))
    deltas = (s[0] - s[2:])
    struct = float(s[0] - s[1])
    order = torch.argsort(deltas, descending=True).tolist()
    positive = [j for j in order if deltas[j] > 0]
    del_masks, keep_masks = [], []
    for K in K_GRID + [len(positive)]:
        dm = torch.ones(E, dtype=torch.bool); km = is_self.clone()
        for j in positive[:K]:
            dm[groups[j]["edges"]] = False; km[groups[j]["edges"]] = True
        del_masks.append(dm); keep_masks.append(km)
    s_del = scores_with_masks(MODELS[MAIN_RUN], data, V_QPROJ[qi], torch.stack(del_masks))
    s_keep = scores_with_masks(MODELS[MAIN_RUN], data, V_QPROJ[qi], torch.stack(keep_masks))
    d_all = float(s[0] - s_del[-1])
    for K, sd, sk in zip(K_GRID, s_del[:-1].tolist(), s_keep[:-1].tolist()):
        b6_rows.append({"val_query": int(qi), "K": K, "n_relations": len(groups), "n_positive": len(positive),
                        "n_used": min(K, len(positive)), "structure": struct, "drop_K": float(s[0]) - sd,
                        "drop_all_positive": d_all,
                        "comprehensiveness": (float(s[0]) - sd) / d_all if d_all > 1e-6 else np.nan,
                        "sufficiency": (sk - float(s[1])) / struct if abs(struct) > 1e-6 else np.nan})
B6 = pd.DataFrame(b6_rows)
B6_summary = (B6.dropna(subset=["comprehensiveness"]).groupby("K")
              .agg(median_comprehensiveness=("comprehensiveness", "median"),
                   mean_comprehensiveness=("comprehensiveness", "mean"),
                   median_sufficiency=("sufficiency", "median"),
                   mean_relations_used=("n_used", "mean"), n_queries=("val_query", "nunique")))
B6_summary["mean_sparsity"] = B6.groupby("K").apply(lambda d: (1 - d.n_used / d.n_relations).mean(),
                                                    include_groups=False)
ok = B6_summary[B6_summary.median_comprehensiveness >= COMPREHENSIVENESS_TARGET]
K_STAR = int(ok.index.min()) if len(ok) else int(max(K_GRID))
share_pos = float((B6.drop_duplicates("val_query").structure > 1e-6).mean())
CONFIG["explainer"] = {"unit": "quan hệ vô hướng (cặp cạnh 2 chiều)", "score": "Δ = S_full − S_without_relation",
                       "K": K_STAR, "chosen_on": f"validation, {B6.val_query.nunique()} truy vấn, "
                       f"trung vị comprehensiveness ≥ {COMPREHENSIVENESS_TARGET}", "share_val_queries_structure_positive": share_pos,
                       "median_positive_relations_per_val_query": float(B6.drop_duplicates("val_query").n_positive.median())}
(OUT / "B1_xai_config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
B6.to_csv(OUT / "B6_val_k_selection_raw.csv", index=False)
B6_summary.to_csv(OUT / "B6_val_k_selection.csv")
print(f"K được chọn trên validation: K = {K_STAR}")
print(f"Tỷ lệ truy vấn val mà cấu trúc đóng góp dương vào điểm của ảnh đúng: {share_pos:.1%}")
B6_summary
