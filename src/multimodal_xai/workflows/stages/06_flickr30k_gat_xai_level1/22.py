r = rank_table.copy()
R_MAIN = f"rank_{MAIN_RUN}"
r["gain"] = r.rank_clip - r[R_MAIN]
succ_pool = r[(r.rank_clip > 1) & (r[R_MAIN] == 1) & (r.rank_no_edge > 1) & (r.rank_random_edges > 1)]
def one_per_image(df):
    return df.drop_duplicates("positive_image_id", keep="first")
success = one_per_image(succ_pool.sort_values(["gain", "caption_id"], ascending=[False, True])).head(5)
used_images = set(success.positive_image_id)
hurt = one_per_image(r[(r.rank_clip == 1) & (r[R_MAIN] > 1)]
                     .sort_values([R_MAIN, "caption_id"], ascending=[False, True]))
hurt = hurt[~hurt.positive_image_id.isin(used_images)]
miss = one_per_image(r[(r.rank_clip > 1) & (r[R_MAIN] > 1)]
                     .sort_values([R_MAIN, "rank_clip", "caption_id"], ascending=[True, False, True]))
n_hurt = min(3, len(hurt))
miss = miss[~miss.positive_image_id.isin(used_images | set(hurt.head(n_hurt).positive_image_id))]
n_miss = min(5 - n_hurt, len(miss)); n_hurt = min(5 - n_miss, len(hurt))
failure = pd.concat([hurt.head(n_hurt).assign(failure_type="đồ thị làm hại"),
                     miss.head(n_miss).assign(failure_type="không cứu được")])
exclude_img = set(success.positive_image_id) | set(failure.positive_image_id)
rand_pool = [i for i in range(len(r)) if r.positive_image_id.iloc[i] not in exclude_img]
_perm = np.random.default_rng(SELECTION_SEED).permutation(rand_pool)
random_q = one_per_image(r.loc[_perm]).head(10)
CASES = pd.concat([success.assign(case="success"), failure.assign(case="failure"),
                   random_q.assign(case="random")]).reset_index(drop=True)
CASES["case_id"] = [f"{c}_{i + 1}" for c, i in zip(CASES.case, CASES.groupby("case").cumcount())]
CASES.to_csv(OUT / "B8_selected_cases.csv", index=False)
print(f"Kho ca thành công do quan hệ thật: {len(succ_pool)} | ca đồ thị làm hại: {len(hurt)} | "
      f"ca không cứu được: {len(miss)}")
CASES[["case_id", "caption_id", "caption", "rank_clip", R_MAIN, "rank_no_edge", "rank_random_edges",
       "failure_type"]]
