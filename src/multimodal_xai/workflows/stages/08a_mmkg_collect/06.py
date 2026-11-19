cov = cov.sort_values(["n_sitelinks", "n_statements", "qid"], ascending=[False, False, True]).reset_index(drop=True)
def entity_kind(q):
    labs = " ".join(lab(c, "en").lower() for c in all_values(ENT[q], "P31"))
    for k in KIND_QUOTA:
        if re.search(rf"\b{k}\b", labs):
            return k
    return ""

cov["kind"] = cov.qid.map(entity_kind)
nat = cov[cov.group == "natural"].head(N_NATURAL_MAX + 5)
stru = cov[cov.group == "structure"].copy()
stru["kind_rank"] = stru.groupby("kind").cumcount()
stru = stru[stru.kind.eq("") | (stru.kind_rank < [math.ceil(1.6 * KIND_QUOTA.get(k, 10 ** 6)) for k in stru.kind])]
stru = stru.head(N_CANDIDATES - len(nat)).drop(columns="kind_rank")
pool = pd.concat([stru, nat]).sort_values(["n_sitelinks", "n_statements", "qid"], ascending=[False, False, True]).reset_index(drop=True)
pool["rank"] = np.arange(len(pool))
print(f"Ứng viên lấy ảnh: {len(pool)} (structure {int((pool.group == 'structure').sum())}, natural {int((pool.group == 'natural').sum())})")
display(pool[["rank", "qid", "label_vi", "group", "n_sitelinks", "n_statements", "p373"]].head(15))
