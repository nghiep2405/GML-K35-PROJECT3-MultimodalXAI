from scipy.stats import wilcoxon

def wil(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[np.isfinite(d) & (d != 0)]
    return float(wilcoxon(d, alternative="greater").pvalue) if len(d) >= 5 else np.nan

M_COLS = ["fid_plus", "fid_minus", "characterization", "fid_perturb", "fid_plus_rand", "fid_plus_deg", "fid_minus_rand",
          "drank_minus_top", "drank_rand", "PN", "PN_rand", "PN_perturb", "PS", "PS_rand", "sparsity_F", "size_S"]
ks = D3[D3.is_K_star]
per_seed = ks.groupby(["dataset", "seed"]).apply(lambda g: pd.Series({
    "n": len(g), "K_star": int(g.K.iloc[0]), **{c: float(g[c].mean()) for c in M_COLS},
    "win_rate": float((g.fid_plus > g.fid_plus_rand).mean()), "wilcoxon_p": wil(g.fid_plus, g.fid_plus_rand),
    "mimic_vs_original_pearson": float(np.corrcoef(g.s_orig, g.s_mimic_full)[0, 1]) if len(g) > 2 else np.nan,
    "returned_is_relevant": float(g.returned_is_relevant.mean())}), include_groups=False).reset_index()
summary = per_seed.drop(columns="seed").groupby("dataset").agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
for name in DATASET_NAMES:
    per_seed[per_seed.dataset == name].to_csv(DATA[name].out / "D3_fidelity_mimic_per_seed.csv", index=False)
    summary.loc[[name]].to_csv(DATA[name].out / "D3_fidelity_mimic_summary.csv")
show = ["n", "K_star", "fid_plus", "fid_plus_rand", "fid_plus_deg", "fid_minus", "fid_minus_rand", "characterization", "fid_perturb",
        "PN", "PN_rand", "PS", "win_rate", "wilcoxon_p"]
display(per_seed[["dataset", "seed"] + show].round(3))
