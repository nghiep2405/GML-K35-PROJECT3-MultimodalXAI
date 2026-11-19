_TEXT_CACHE = {}
from multimodal_xai.workflows.runner import run_shared
run_shared('ranking', globals())





DATA_BY_KIND = {DATA[n].kind: DATA[n] for n in DATA}
MAX_TEST = int(os.environ.get("XAI2_MAX_TEST", "0"))
TEST = {}
for name in DATASET_NAMES:
    q = DATA[name].test_q if not MAX_TEST else DATA[name].test_q.iloc[:MAX_TEST]
    t0 = time.time(); TEST[name] = Split(name, "test", q.reset_index(drop=True))
    print(f"{name}: CLIP top-{TOPK} cho {len(q)} truy vấn test ({time.time() - t0:.0f}s)")

RES, C1_rows = {}, []
for name in DATASET_NAMES:
    D, Sp = DATA[name], TEST[name]
    for s in SEEDS:
        sc = SCORERS[(name, s)]; rec = []
        with torch.no_grad():
            for i in range(len(Sp.q)):
                c = Sp.ctx(i, sc); order = top_order(c)
                top = int(order[0]); rec.append(dict(query_id=c["q"].query_id, direction=c["q"].prediction_direction,
                    clip_rank=c["base"], hybrid_rank=rank_in(c), top1_row=int(c["rows"][top]),
                    top1_entity=int(c["ent"][top]) if c["ent"] is not None else int(D.img_ent[int(c["rows"][top]), 0]),
                    target_in_topk=bool(c["tmask"].any())))
        R = pd.DataFrame(rec); RES[(name, s)] = R
        saved_dir = latest_locked(D.locked_dir[s], D.ckpt[s])
        if saved_dir is not None:
            ref = pd.read_csv(saved_dir / "test_query_ranks.csv").set_index("query_id").reindex(R.query_id)
            same_clip = float((ref.clip_rank.to_numpy() == R.clip_rank.to_numpy()).mean())
            same_hyb = float((ref.hybrid_rank.to_numpy() == R.hybrid_rank.to_numpy()).mean())
            mrr_ref = float((1 / ref.hybrid_rank).mean())
        else:
            same_clip = same_hyb = mrr_ref = float("nan")
        C1_rows.append(dict(dataset=name, seed=s, epoch=sc.epoch, alpha_clip=sc.alpha, n=len(R),
                            MRR_clip=float((1 / R.clip_rank).mean()), MRR_hybrid=float((1 / R.hybrid_rank).mean()),
                            MRR_hybrid_nhom=mrr_ref, khop_clip_rank=same_clip, khop_hybrid_rank=same_hyb,
                            thu_muc_nhom=str(saved_dir.name) if saved_dir else None))
        R.to_csv(D.out / f"C1_test_ranks_seed{s}.csv", index=False)
C1 = pd.DataFrame(C1_rows)
for name in DATASET_NAMES:
    C1[C1.dataset == name].to_csv(DATA[name].out / "C1_reproduction.csv", index=False)
display(C1)
