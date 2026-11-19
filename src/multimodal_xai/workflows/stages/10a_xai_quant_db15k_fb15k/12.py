D3_parts = {}
for name in DATASET_NAMES:
    D, Sp = DATA[name], TEST[name]
    for s in SEEDS:
        sc = SCORERS[(name, s)]; cache = D.out / f"_D3_seed{s}.csv"
        if sc.alpha == 1.0:
            print(f"{name} seed {s}: α_clip = 1, đồ thị không tham gia xếp hạng → bỏ qua"); continue
        if REUSE_DONE and cache.exists():
            D3_parts[(name, s)] = pd.read_csv(cache); print(f"{name} seed {s}: dùng lại kết quả đã có"); continue
        Ks = K_GRID if s == MAIN_SEED else [K_STAR[name]]
        rows, t0 = [], time.time()
        for j, sel in enumerate(SEL[name].itertuples()):
            c = Sp.ctx(int(sel.qi), sc); pos = int(top_order(c)[0])
            meta, rs = explain_quant(c, D, sc, pos, Ks, N_RANDOM_L3, K_STAR[name])
            for r in rs:
                rows.append(dict(dataset=name, seed=s, query_id=c["q"].query_id, in_C9=bool(sel.in_C9), in_C3=bool(sel.in_C3),
                                 direction=c["q"].prediction_direction, relation=D.rel_label[int(c["q"].relation_index)],
                                 returned_is_relevant=bool(c["tmask"][pos]), entity=meta["entity"], entity_label=D.ent_label[meta["entity"]],
                                 n_facts=meta["n_facts"], n_F=meta["n_F"], n_cand=meta["n_cand"], s_orig=meta["s_orig"],
                                 s_mimic_full=meta["s_mimic_full"], is_K_star=r["K"] == K_STAR[name], **r))
            if j % 10 == 9:
                print(f"  {name} seed {s}: {j + 1}/{len(SEL[name])} truy vấn, {time.time() - t0:.0f}s")
        part = pd.DataFrame(rows); part.to_csv(cache, index=False); D3_parts[(name, s)] = part
        print(f"{name} seed {s}: xong {part.query_id.nunique()} giải thích trong {time.time() - t0:.0f}s")
D3 = pd.concat(D3_parts.values(), ignore_index=True)
for name in DATASET_NAMES:
    D3[D3.dataset == name].to_csv(DATA[name].out / "D3_fidelity_mimic_per_query.csv", index=False)
