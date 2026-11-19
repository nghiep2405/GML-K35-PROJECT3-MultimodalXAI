C9_rows, C9_reused = [], []
for name in DATASET_NAMES:
    D, Sp = DATA[name], TEST[name]; K = K_STAR[name]
    if REUSE_DONE and (D.out / "C9_fidelity_per_query.csv").exists():
        C9_reused.append(pd.read_csv(D.out / "C9_fidelity_per_query.csv")); print(f"{name}: dùng lại C9 đã có"); continue
    ind = np.where(Sp.q.prediction_direction.ne("direct").to_numpy())[0]
    pick_i = np.sort(np.random.default_rng(SELECTION_SEED).choice(ind, min(N_FID, len(ind)), replace=False))
    for s in SEEDS:
        sc = SCORERS[(name, s)]
        if sc.alpha == 1.0:
            print(f"{name} seed {s}: α_clip = 1, đồ thị không tham gia xếp hạng → bỏ qua"); continue
        for i in pick_i:
            c = Sp.ctx(int(i), sc); p = int(top_order(c)[0])
            em = explain_modality(c, D, sc, p); ef = explain_facts(c, D, sc, p, [K])
            row = dict(dataset=name, seed=s, query_id=c["q"].query_id, n_facts=ef["n_facts"],
                       struct_w=em["weights"][0], struct_delta=em["drop"].get(0, {}).get("delta", np.nan),
                       struct_rank=em["drop"].get(0, {}).get("rank", np.nan))
            if ef["ok"] and f"top@{K}" in ef["removal"]:
                t, r = ef["removal"][f"top@{K}"], ef["removal"].get(f"rand@{K}", {"dscore": np.nan, "drank": np.nan})
                row.update(e3_top_dscore=t["dscore"], e3_top_drank=t["drank"], e3_rand_dscore=r["dscore"], e3_rand_drank=r["drank"],
                           mimic_s_full=ef["s_full"], s_orig=ef["s_orig"], rank_full=ef["rank_full"])
            C9_rows.append(row)
C9 = pd.concat([pd.DataFrame(C9_rows)] + C9_reused, ignore_index=True)
per_seed = C9.groupby(["dataset", "seed"]).agg(
    n=("query_id", "size"), co_bang_chung=("e3_top_dscore", "count"),
    e3_top_dscore=("e3_top_dscore", "mean"), e3_rand_dscore=("e3_rand_dscore", "mean"),
    e3_top_drank=("e3_top_drank", "mean"), e3_rand_drank=("e3_rand_drank", "mean"),
    win_rate=("e3_top_dscore", lambda x: float((x > C9.loc[x.index, "e3_rand_dscore"]).mean())),
    struct_w=("struct_w", "mean"), struct_delta=("struct_delta", "mean"),
    mat_hang1_khi_che_cau_truc=("struct_rank", lambda x: float((x > 1).mean()))).reset_index()
C9_summary = per_seed.groupby("dataset").agg(["mean", "std"]).drop(columns="seed", level=0)
for name in DATASET_NAMES:
    D = DATA[name]
    C9[C9.dataset == name].to_csv(D.out / "C9_fidelity_per_query.csv", index=False)
    per_seed[per_seed.dataset == name].to_csv(D.out / "C9_fidelity_per_seed.csv", index=False)
    if name in C9_summary.index:
        C9_summary.loc[[name]].to_csv(D.out / "C9_fidelity_mean_std.csv")
display(per_seed.round(4)); display(C9_summary.round(4))
