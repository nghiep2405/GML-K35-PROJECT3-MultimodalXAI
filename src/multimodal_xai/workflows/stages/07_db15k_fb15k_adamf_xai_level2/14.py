K_STAR, VAL = {}, {}
for name in DATASET_NAMES:
    D, sc = DATA[name], SCORERS[(name, MAIN_SEED)]
    if REUSE_DONE and (D.out / "C6_K_star.json").exists():
        info = json.load(open(D.out / "C6_K_star.json")); K_STAR[name] = int(info["K_star"])
        print(f"{name}: dùng lại C6 đã có → K* = {K_STAR[name]} | trung vị độ đầy đủ {info['median_comprehensiveness']} | "
              f"tương quan học lại–gốc = {info['mimic_full_vs_original_pearson']:.3f}"); continue
    vq = D.val_q[D.val_q.prediction_direction != "direct"]
    vq = vq.iloc[np.sort(np.random.default_rng(SELECTION_SEED).choice(len(vq), min(N_VAL_K, len(vq)), replace=False))].reset_index(drop=True)
    VAL[name] = Split(name, "val", vq); rows = []
    for i in range(len(vq)):
        c = VAL[name].ctx(i, sc); pos = int(top_order(c)[0])
        ex = explain_facts(c, D, sc, pos, K_GRID, n_random=0)
        if not ex["ok"] or ex["d_all"] <= 1e-9:
            continue
        rows.append(dict(query_id=vq.query_id[i], s_orig=ex["s_orig"], s_full=ex["s_full"], d_all=ex["d_all"],
                         **{f"comp@{K}": ex["removal"].get(f"top@{K}", {"dscore": 0.})["dscore"] / ex["d_all"] for K in K_GRID}))
    V6 = pd.DataFrame(rows); V6.to_csv(D.out / "C6_choose_K_validation.csv", index=False)
    med = {K: float(V6[f"comp@{K}"].median()) for K in K_GRID} if len(V6) else {}
    K_STAR[name] = next((K for K in K_GRID if med.get(K, 0) >= COMPREHENSIVENESS_TARGET), K_GRID[-1])
    corr = float(np.corrcoef(V6.s_orig, V6.s_full)[0, 1]) if len(V6) > 2 else float("nan")
    json.dump(dict(K_star=K_STAR[name], median_comprehensiveness=med, n=len(V6), mimic_full_vs_original_pearson=corr,
                   mimic=MIMIC, M=M_CAND_FACTS), open(D.out / "C6_K_star.json", "w"), indent=2, ensure_ascii=False)
    print(f"{name}: K* = {K_STAR[name]} | trung vị độ đầy đủ {med} | n = {len(V6)} | tương quan học lại–gốc = {corr:.3f}")
