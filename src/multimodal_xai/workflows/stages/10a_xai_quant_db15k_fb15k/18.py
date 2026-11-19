val = ks.groupby(["dataset", "seed"]).agg(n=("query_id", "size"), PN=("PN", "mean"), PN_rand=("PN_rand", "mean"),
                                          PN_perturb=("PN_perturb", "mean"), PS=("PS", "mean"),
                                          in_train=("in_train", "mean"), heldout_overlap=("heldout_overlap", "sum"),
                                          on_path=("on_path", "mean")).reset_index()
val["structural_i_ok"] = (val.in_train == 1.0) & (val.heldout_overlap == 0)
for name in DATASET_NAMES:
    val[val.dataset == name].to_csv(DATA[name].out / "D6_validity.csv", index=False)
display(val.round(3))
assert val.structural_i_ok.all(), "Có giải thích chứa bộ ba ngoài train hoặc trùng val/test — kiểm tra lại dữ liệu."

run_info = dict(notebook="10a_xai_quant_db15k_fb15k", K_star=K_STAR, seeds=SEEDS, main_seed=MAIN_SEED, n_random=N_RANDOM_L3,
                mimic=MIMIC, M=M_CAND_FACTS, K_grid=K_GRID, topk=TOPK, selection_seed=SELECTION_SEED,
                n_queries={n: int(len(SEL[n])) for n in DATASET_NAMES})
for name in DATASET_NAMES:
    json.dump(run_info, open(DATA[name].out / "10a_run_info.json", "w"), indent=2, ensure_ascii=False)
print("Xong. Kết quả trong", {n: str(DATA[n].out) for n in DATASET_NAMES})
