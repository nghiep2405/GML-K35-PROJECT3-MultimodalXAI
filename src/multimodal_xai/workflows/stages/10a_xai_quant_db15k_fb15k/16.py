cur = D3[D3.seed == MAIN_SEED].groupby(["dataset", "K"]).agg(
    n=("query_id", "size"), size_S=("size_S", "mean"), sparsity_F=("sparsity_F", "mean"), sparsity_cand=("sparsity_cand", "mean"),
    fid_plus=("fid_plus", "mean"), fid_plus_rand=("fid_plus_rand", "mean"), fid_minus=("fid_minus", "mean"),
    fid_minus_rand=("fid_minus_rand", "mean"), characterization=("characterization", "mean"), PN=("PN", "mean"), PS=("PS", "mean")).reset_index()
cur["is_K_star"] = [K == K_STAR[d] for d, K in zip(cur.dataset, cur.K)]
for name in DATASET_NAMES:
    cur[cur.dataset == name].to_csv(DATA[name].out / "D5_sparsity_K_curve.csv", index=False)
display(cur.round(3))
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(DATASET_NAMES), figsize=(5.2 * len(DATASET_NAMES), 3.6), squeeze=False)
for ax, name in zip(axes[0], DATASET_NAMES):
    c = cur[cur.dataset == name]
    ax.plot(c.K, c.fid_plus, "o-", label="Fid+ (bỏ S)"); ax.plot(c.K, c.fid_plus_rand, "o--", label="Fid+ ngẫu nhiên")
    ax.plot(c.K, c.fid_minus, "s-", label="Fid− (chỉ giữ S)"); ax.plot(c.K, c.sparsity_F, "^:", label="sparsity")
    ax.axvline(K_STAR[name], color="gray", lw=0.8); ax.set_xticks(K_GRID); ax.set_xlabel("K"); ax.set_title(f"{name}, seed {MAIN_SEED}")
axes[0][0].legend(fontsize=8); plt.tight_layout()
for name in DATASET_NAMES:
    fig.savefig(DATA[name].out / "D5_K_curve.png", dpi=150)
plt.show()
