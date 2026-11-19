def build_benchmark_q(q):
    q = q.copy()
    if "is_test_cold_start" in q.columns:          # (needs_human_review = True cho mọi truy vấn gián tiếp sinh theo mẫu nên không lọc cột đó)
        q = q[~q["is_test_cold_start"].fillna(False).astype(bool)]
    q = q.drop_duplicates("target_entity_index")
    rng = np.random.default_rng(SELECTION_SEED); parts = []
    for d, n in [("direct", N_BENCH_DIRECT), ("head", N_BENCH_HEAD), ("tail", N_BENCH_TAIL)]:
        cand = q[q.prediction_direction == d]
        parts.append(cand.iloc[np.sort(rng.choice(len(cand), min(n, len(cand)), replace=False))])
    return pd.concat(parts)

TEST, SEL, D1_rows = {}, {}, []
for name in DATASET_NAMES:
    D = DATA[name]; tq = D.test_q.reset_index(drop=True)
    ind = np.where(tq.prediction_direction.ne("direct").to_numpy())[0]
    c9 = np.sort(np.random.default_rng(SELECTION_SEED).choice(ind, min(N_FID, len(ind)), replace=False))
    B = build_benchmark_q(tq)
    src = pd.DataFrame({"query_id": tq.query_id, "in_C9": False})
    src.loc[c9, "in_C9"] = True
    src["in_C3"] = src.query_id.isin(set(B.loc[B.prediction_direction != "direct", "query_id"]))
    src = src[src.in_C9 | src.in_C3]
    # CLIP top-k trên TOÀN BỘ test như notebook 07 (cùng kích thước lô → tích vô hướng trùng từng bit), rồi chỉ dùng các truy vấn đã chọn
    t0 = time.time(); TEST[name] = Split(name, "test", tq); SEL[name] = src.rename_axis("qi").reset_index()
    print(f"{name}: {len(src)} truy vấn (C9 {int(src.in_C9.sum())}, C3 {int(src.in_C3.sum())}, chung {int((src.in_C9 & src.in_C3).sum())}) "
          f"| CLIP top-{TOPK} cho {len(tq)} truy vấn test: {time.time() - t0:.0f}s")
    for s in SEEDS:
        sc = SCORERS[(name, s)]; Sp = TEST[name]
        with torch.no_grad():
            mine = pd.DataFrame([dict(query_id=Sp.q.query_id[i], clip_rank=(c := Sp.ctx(i, sc))["base"], hybrid_rank=rank_in(c))
                                 for i in SEL[name].qi])
        ref_p = L2_XAI(name) / f"C1_test_ranks_seed{s}.csv"
        ref_dir = latest_locked(D.locked_dir[s], D.ckpt[s]) if not ref_p.exists() else None
        ref = pd.read_csv(ref_p) if ref_p.exists() else (pd.read_csv(ref_dir / "test_query_ranks.csv") if ref_dir is not None else None)
        if ref is not None:
            ref = ref.set_index("query_id").reindex(mine.query_id)
            D1_rows.append(dict(dataset=name, seed=s, n=len(mine), reference=ref_p.name if ref_p.exists() else f"{ref_dir.name}/test_query_ranks.csv",
                                match_clip_rank=float((ref.clip_rank.to_numpy() == mine.clip_rank.to_numpy()).mean()),
                                match_hybrid_rank=float((ref.hybrid_rank.to_numpy() == mine.hybrid_rank.to_numpy()).mean())))
        else:
            D1_rows.append(dict(dataset=name, seed=s, n=len(mine), reference=None, match_clip_rank=np.nan, match_hybrid_rank=np.nan))
D1 = pd.DataFrame(D1_rows)
for name in DATASET_NAMES:
    D1[D1.dataset == name].to_csv(DATA[name].out / "D1_replay_check.csv", index=False)
display(D1)
if (D1.match_hybrid_rank < 0.99).any():
    print("Lưu ý: có seed khớp hạng < 99% với cấp 2. Chênh lệch nhỏ có thể do tính CLIP theo lô khác; số liệu bên dưới vẫn đo trên đúng checkpoint.")
