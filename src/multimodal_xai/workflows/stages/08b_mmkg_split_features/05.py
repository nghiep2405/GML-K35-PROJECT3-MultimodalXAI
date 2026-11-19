rng = np.random.default_rng(SPLIT_SEED)
trip = trip[["head", "relation", "tail", "head_qid", "pid", "tail_qid", "source", "statement_id", "retrieved_at"]].copy()
trip["unit"] = [f"{min(h, t)}|{r}|{max(h, t)}" if p in SYMMETRIC else f"{h}|{r}|{t}"
                for h, r, t, p in zip(trip["head"], trip.relation, trip["tail"], trip.pid)]
units = trip.groupby("unit").agg(relation=("relation", "first")).reset_index()
units["split"] = "train"
for r, g in units.groupby("relation"):
    idx = g.index.to_numpy().copy(); rng.shuffle(idx)
    n = len(idx); n_te = int(round(n * SPLIT_FRAC[2])); n_va = int(round(n * SPLIT_FRAC[1]))
    if n < 4:                                   # quan hệ quá ít triple: giữ hết trong train
        continue
    units.loc[idx[:n_te], "split"] = "test"; units.loc[idx[n_te:n_te + n_va], "split"] = "valid"
trip["split"] = trip.unit.map(units.set_index("unit").split)
for it in range(20):
    train_ents = set(trip.loc[trip.split == "train", "head"]) | set(trip.loc[trip.split == "train", "tail"])
    bad_u = set(trip.loc[(trip.split != "train") & ~(trip["head"].isin(train_ents) & trip["tail"].isin(train_ents)), "unit"])
    if not bad_u:
        break
    trip.loc[trip.unit.isin(bad_u), "split"] = "train"
key = lambda df: set(zip(df["head"], df.relation, df["tail"]))
TR, VA, TE = (trip[trip.split == s] for s in ["train", "valid", "test"])
assert not (key(TE) & key(TR)) and not (key(VA) & key(TR)) and not (key(TE) & key(VA))
rev = lambda df: {(t, r, h) for h, r, t, p in zip(df["head"], df.relation, df["tail"], df.pid) if p in SYMMETRIC}
assert not (rev(TE) & key(TR)) and not (rev(VA) & key(TR)), "Chiều ngược của triple đối xứng val/test nằm trong train"
assert set(TE["head"]) | set(TE["tail"]) <= set(TR["head"]) | set(TR["tail"])
split_tab = trip.groupby(["pid", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "valid", "test"], fill_value=0)
print(f"Triple: train {len(TR)}, valid {len(VA)}, test {len(TE)} (sau {it} vòng chuyển về train)")
display(split_tab.T)
