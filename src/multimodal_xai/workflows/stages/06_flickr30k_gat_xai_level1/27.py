B10_rows = []
for name in ("seed_2026", "seed_2027", "seed_2028"):
    model, graphs, a = MODELS[name], test_graphs_for(name), ALPHA[name]
    for row in CASES.itertuples():
        qi, img = int(row.query_index), int(T_TARGET[row.query_index])
        data, q = graphs[img], QPROJ_TEST[name][qi]
        ex = explain(model, data, q, a, K_STAR)
        k = ex["n_explanation"]
        base_rank = int(1 + (FUSED_TEST[name][qi] > FUSED_TEST[name][qi, img]).sum())
        rec = {"seed_run": name, "case_id": row.case_id, "case": row.case, "n_relations": ex["n_relations"],
               "k_used": k, "S_full": ex["S_full"], "structure": ex["structure_contribution"],
               "drop_graph_exp": ex["S_full"] - ex["S_delete_explanation"],
               "rank_base": base_rank,
               "rank_after_exp": fused_rank_after(name, qi, img, ex["S_delete_explanation"]),
               "sufficiency": ex["sufficiency"], "comprehensiveness": ex["comprehensiveness"],
               "n_spatial_in_exp": ex["n_spatial_in_explanation"],
               "top_spatial_delta": ex["top_spatial"]["delta_graph"] if ex["top_spatial"] else np.nan,
               "top_star_delta": max([d for d, g in zip(ex["_deltas"], ex["_groups"]) if g["kind"] == "star"],
                                     default=np.nan)}
        if k == 0 or k >= ex["n_relations"]:
            rec.update(drop_graph_rand=np.nan, rank_after_rand=np.nan)
        else:
            groups = ex["_groups"]
            E = data.edge_index.shape[1]
            g_rng = np.random.default_rng(SELECTION_SEED + qi)
            masks = []
            for _ in range(N_RANDOM_REPEATS_B10):
                pick = g_rng.choice(len(groups), k, replace=False)
                m = torch.ones(E, dtype=torch.bool)
                for j in pick:
                    m[groups[j]["edges"]] = False
                masks.append(m)
            s_rand = scores_with_masks(model, data, q, torch.stack(masks))
            rec["drop_graph_rand"] = float((ex["S_full"] - s_rand).mean())
            rec["rank_after_rand"] = float(np.mean([fused_rank_after(name, qi, img, float(v)) for v in s_rand]))
        rec["drop_fused_exp"] = a * rec["drop_graph_exp"]
        rec["drop_fused_rand"] = a * rec["drop_graph_rand"] if not np.isnan(rec["drop_graph_rand"]) else np.nan
        rec["exp_beats_random"] = (rec["drop_graph_exp"] > rec["drop_graph_rand"]) if not np.isnan(
            rec["drop_graph_rand"]) else np.nan
        B10_rows.append(rec)
B10 = pd.DataFrame(B10_rows)
B10.to_csv(OUT / "B10_deletion_check_raw.csv", index=False)

valid = B10.dropna(subset=["drop_graph_rand"])
per_seed = valid.groupby("seed_run").agg(
    n_queries=("case_id", "count"),
    drop_graph_exp=("drop_graph_exp", "mean"), drop_graph_rand=("drop_graph_rand", "mean"),
    drop_fused_exp=("drop_fused_exp", "mean"), drop_fused_rand=("drop_fused_rand", "mean"),
    rank_change_exp=("rank_after_exp", lambda s: (s - valid.loc[s.index, "rank_base"]).mean()),
    rank_change_rand=("rank_after_rand", lambda s: (s - valid.loc[s.index, "rank_base"]).mean()),
    win_rate=("exp_beats_random", "mean"), sufficiency=("sufficiency", "median"),
    comprehensiveness=("comprehensiveness", "median"),
    share_spatial_in_exp=("n_spatial_in_exp", lambda s: s.sum() / max(1, valid.loc[s.index, "k_used"].sum())),
    top_spatial_delta=("top_spatial_delta", "mean"), top_star_delta=("top_star_delta", "mean"))
agg = per_seed.agg(["mean", "std"]).T
agg["báo cáo (mean ± SD, 3 seed)"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(agg["mean"], agg["std"])]
per_seed.to_csv(OUT / "B10_per_seed.csv")
agg.to_csv(OUT / "B10_summary_mean_sd.csv")
(OUT / "B10_summary.json").write_text(json.dumps({
    "K": K_STAR, "n_random_repeats": N_RANDOM_REPEATS_B10,
    "queries": "5 success + 5 failure + 10 random (B8)",
    "excluded_queries_without_positive_relation_or_all_relations": int(B10.drop_graph_rand.isna().sum()),
    "mean_sd_over_3_seeds": {k: {"mean": float(v["mean"]), "sd": float(v["std"])} for k, v in agg.iterrows()},
}, indent=2, ensure_ascii=False), encoding="utf-8")
agg[["báo cáo (mean ± SD, 3 seed)"]]
