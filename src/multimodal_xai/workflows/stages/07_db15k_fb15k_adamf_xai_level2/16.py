def pick(df, n, rng, used_rel, used_ent):
    out = []
    for i in rng.permutation(len(df)):
        r = df.iloc[i]
        if r.relation_index in used_rel or r.target_entity_index in used_ent:
            continue
        out.append(r); used_rel.add(r.relation_index); used_ent.add(r.target_entity_index)
        if len(out) == n:
            break
    return out

CASES = {}
for name in DATASET_NAMES:
    D = DATA[name]; R = RES[(name, MAIN_SEED)].merge(TEST[name].q[["query_id", "query_text", "anchor_entity_index", "relation_index",
                                                                   "target_entity_index"]], on="query_id")
    R["qi"] = np.arange(len(R)); ind = R[R.direction != "direct"]
    rng = np.random.default_rng(SELECTION_SEED); ur, ue = set(), set()
    succ = pick(ind[(ind.clip_rank >= 20) & (ind.hybrid_rank == 1)], 5, rng, ur, ue)
    hurt = pick(ind[(ind.clip_rank <= 5) & (ind.hybrid_rank > ind.clip_rank)], 3, rng, ur, ue)
    miss = pick(ind[ind.target_in_topk & (ind.hybrid_rank >= 100)], 5 - len(hurt), rng, ur, ue)
    if len(hurt) + len(miss) < 5:
        hurt += pick(ind[ind.hybrid_rank > ind.clip_rank], 5 - len(hurt) - len(miss), rng, ur, ue)
    rows = [("thanh_cong", r) for r in succ] + [("that_bai_bi_lam_hai", r) for r in hurt] + [("that_bai_khong_cuu_duoc", r) for r in miss]
    CASES[name] = pd.DataFrame([dict(case=c, **r.to_dict()) for c, r in rows])
    CASES[name].to_csv(D.out / "C7_cases.csv", index=False)
    display(CASES[name][["case", "query_text", "clip_rank", "hybrid_rank"]])
