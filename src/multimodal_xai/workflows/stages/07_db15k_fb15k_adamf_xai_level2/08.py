C2_rows = []
for (name, s), R in RES.items():
    for g, m in [("tất cả", np.ones(len(R), bool)), ("trực tiếp", R.direction.eq("direct")), ("gián tiếp-head", R.direction.eq("head")),
                 ("gián tiếp-tail", R.direction.eq("tail"))]:
        x = R[m]
        C2_rows.append(dict(dataset=name, seed=s, nhom=g, n=len(x), tot_hon=int((x.hybrid_rank < x.clip_rank).sum()),
                            te_hon=int((x.hybrid_rank > x.clip_rank).sum()), giu_nguyen=int((x.hybrid_rank == x.clip_rank).sum()),
                            cuu_len_hang1=int(((x.clip_rank > 1) & (x.hybrid_rank == 1)).sum()),
                            mat_hang1=int(((x.clip_rank == 1) & (x.hybrid_rank > 1)).sum()),
                            MRR_clip=float((1 / x.clip_rank).mean()), MRR_hybrid=float((1 / x.hybrid_rank).mean())))
C2 = pd.DataFrame(C2_rows)
C2_summary = C2.groupby(["dataset", "nhom"]).agg(n=("n", "first"), MRR_clip=("MRR_clip", "mean"), MRR_hybrid_mean=("MRR_hybrid", "mean"),
                                                  MRR_hybrid_std=("MRR_hybrid", "std"), cuu_len_hang1=("cuu_len_hang1", "mean"),
                                                  mat_hang1=("mat_hang1", "mean"), tot_hon=("tot_hon", "mean"), te_hon=("te_hon", "mean")).reset_index()
for name in DATASET_NAMES:
    C2[C2.dataset == name].to_csv(DATA[name].out / "C2_counts_by_seed.csv", index=False)
    C2_summary[C2_summary.dataset == name].to_csv(DATA[name].out / "C2_counts_mean_std.csv", index=False)
display(C2_summary.round(4))
