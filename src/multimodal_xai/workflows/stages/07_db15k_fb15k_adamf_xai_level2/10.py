def build_benchmark(name):
    q = TEST[name].q.copy()
    if "is_test_cold_start" in q.columns:          # (needs_human_review = True cho mọi truy vấn gián tiếp sinh theo mẫu nên không lọc cột đó)
        q = q[~q["is_test_cold_start"].fillna(False).astype(bool)]
    q = q.drop_duplicates("target_entity_index")
    rng = np.random.default_rng(SELECTION_SEED); parts = []
    for d, n in [("direct", N_BENCH_DIRECT), ("head", N_BENCH_HEAD), ("tail", N_BENCH_TAIL)]:
        cand = q[q.prediction_direction == d]
        parts.append(cand.iloc[np.sort(rng.choice(len(cand), min(n, len(cand)), replace=False))])
    return pd.concat(parts)

BENCH, C3_rows = {}, []
for name in DATASET_NAMES:
    B = build_benchmark(name); BENCH[name] = B
    for s in SEEDS:
        R = RES[(name, s)].set_index("query_id").loc[B.query_id]
        for g, m in [("trực tiếp", B.prediction_direction.eq("direct").to_numpy()), ("gián tiếp", B.prediction_direction.ne("direct").to_numpy()),
                     ("tất cả", np.ones(len(B), bool))]:
            for meth, col in [("CLIP", "clip_rank"), ("CLIP + AdaMF", "hybrid_rank")]:
                r = R[col].to_numpy()[m]
                C3_rows.append(dict(dataset=name, seed=s, nhom=g, phuong_phap=meth, n=int(m.sum()), MRR=float((1 / r).mean()),
                                    R1=float((r <= 1).mean()), R5=float((r <= 5).mean()), R10=float((r <= 10).mean())))
    detail = B[["query_id", "query_text", "prediction_direction"]].copy()
    for s in SEEDS:
        R = RES[(name, s)].set_index("query_id").loc[B.query_id]
        detail[f"clip_rank"] = R.clip_rank.to_numpy(); detail[f"hybrid_rank_seed{s}"] = R.hybrid_rank.to_numpy()
    detail.to_csv(DATA[name].out / "C3_benchmark_queries.csv", index=False)
C3 = pd.DataFrame(C3_rows)
C3_summary = C3.groupby(["dataset", "nhom", "phuong_phap"]).agg(n=("n", "first"), MRR=("MRR", "mean"), MRR_std=("MRR", "std"),
                                                                  R1=("R1", "mean"), R10=("R10", "mean")).reset_index()
for name in DATASET_NAMES:
    C3_summary[C3_summary.dataset == name].to_csv(DATA[name].out / "C3_benchmark_metrics.csv", index=False)
display(C3_summary.round(3))
