K_STAR = {}
for name in DATASET_NAMES:
    D, sc = DATA[name], SCORERS[(name, MAIN_SEED)]
    p = L2_XAI(name) / "C6_K_star.json"
    if p.exists():
        info = json.load(open(p)); K_STAR[name] = int(info["K_star"])
        print(f"{name}: K* = {K_STAR[name]} (từ C6 cấp 2, n = {info.get('n')}, trung vị độ đầy đủ {info.get('median_comprehensiveness')})")
        continue
    vq = D.val_q[D.val_q.prediction_direction != "direct"]
    vq = vq.iloc[np.sort(np.random.default_rng(SELECTION_SEED).choice(len(vq), min(N_VAL_K, len(vq)), replace=False))].reset_index(drop=True)
    V = Split(name, "val", vq); rows = []
    for i in range(len(vq)):
        c = V.ctx(i, sc); pos = int(top_order(c)[0])
        ex = explain_facts(c, D, sc, pos, K_GRID, n_random=0)
        if not ex["ok"] or ex["d_all"] <= 1e-9:
            continue
        rows.append({f"comp@{K}": ex["removal"].get(f"top@{K}", {"dscore": 0.})["dscore"] / ex["d_all"] for K in K_GRID})
    V6 = pd.DataFrame(rows); med = {K: float(V6[f"comp@{K}"].median()) for K in K_GRID} if len(V6) else {}
    K_STAR[name] = next((K for K in K_GRID if med.get(K, 0) >= COMPREHENSIVENESS_TARGET), K_GRID[-1])
    json.dump(dict(K_star=K_STAR[name], median_comprehensiveness=med, n=len(V6), recomputed_in="10a"), open(D.out / "C6_K_star_recomputed.json", "w"), indent=2)
    print(f"{name}: K* = {K_STAR[name]} (tính lại trên validation, n = {len(V6)})")
