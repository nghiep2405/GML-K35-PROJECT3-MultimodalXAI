import torch
def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# OpenKE (giống reproduction/ của DB15K): dòng đầu là số dòng; triple ghi "head tail relation"
def write_map(path, names):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(names)}\n"); f.writelines(f"{n}\t{i}\n" for i, n in enumerate(names))
write_map(REPRO / "entity2id.txt", ents.qid.tolist()); write_map(REPRO / "relation2id.txt", rels.pid.tolist())
for name, df in [("train2id.txt", TR), ("valid2id.txt", VA), ("test2id.txt", TE)]:
    with open(REPRO / name, "w") as f:
        f.write(f"{len(df)}\n"); f.writelines(f"{h} {t} {r}\n" for h, t, r in zip(df["head"], df["tail"], df.relation))
trip.drop(columns="unit").to_csv(PROC / "triples_split.tsv", sep="\t", index=False)

ents_out = ents.rename(columns={"label_vi": "entity_label"}).assign(entity_label_en=ents.label_en,
            entity_description=ent_text, num_images=ents.entity_index.map(imgs.groupby("entity_index").size()).fillna(0).astype(int))
ents_out["has_image"] = ents_out.num_images > 0
ents_out.to_parquet(PROC / "entities.parquet", index=False)
Q.to_parquet(PROC / "queries.parquet", index=False)
REL.to_parquet(PROC / "relevance.parquet", index=False)
FLT.to_parquet(PROC / "query_filter.parquet", index=False)
pools = pd.concat([pd.DataFrame({"pool_id": f"mmkg_{s}_all_images", "split": s, "image_id": POOL[s]}) for s in ["val", "test"]])
pools["entity_index"] = pools.image_id.map(imgs.set_index("image_id").entity_index)
pools.to_parquet(PROC / "candidate_pools.parquet", index=False)
imgs[["image_id", "feature_row", "entity_index", "retrieval_split"]].to_parquet(PROC / "image_feature_index.parquet", index=False)
imgs[["image_id", "entity_index", "link_method"]].rename(columns={"link_method": "link_source"}).to_parquet(PROC / "entity_image.parquet", index=False)
imgs.rename(columns={"file": "image_path"}).to_parquet(PROC / "images.parquet", index=False)
np.save(PROC / "clip_image_features.npy", img_feat)
torch.save(torch.from_numpy(txt_feat), PROC / "textual.pt")
(PROC / "clip_feature_metadata.json").write_text(json.dumps(dict(image_encoder=IMG_MODEL, text_encoder=TXT_MODEL, dim=512,
    normalized="L2", entity_text="label_vi. desc_vi. wiki_intro_vi", visual_policy="train_image_CLIP_mean_L2 (tính ở 09)"),
    ensure_ascii=False, indent=2), encoding="utf-8")

# mã tạo split độc lập (nộp kèm, chạy lại cho đúng các file train/valid/test)
make_splits = '''"""make_splits.py — tái tạo train/valid/test của ViHeritage-MMKG từ processed/triples_all.tsv.
Dùng: python make_splits.py <thư mục data/MMKG>"""
import sys, numpy as np, pandas as pd
from pathlib import Path
SPLIT_SEED, SPLIT_FRAC, SYMMETRIC = %d, %r, %r
root = Path(sys.argv[1]); trip = pd.read_csv(root / "processed/triples_all.tsv", sep="\\t", keep_default_na=False)
rng = np.random.default_rng(SPLIT_SEED)
trip["unit"] = [f"{min(h, t)}|{r}|{max(h, t)}" if p in SYMMETRIC else f"{h}|{r}|{t}" for h, r, t, p in zip(trip["head"], trip.relation, trip["tail"], trip.pid)]
units = trip.groupby("unit").agg(relation=("relation", "first")).reset_index(); units["split"] = "train"
for r, g in units.groupby("relation"):
    idx = g.index.to_numpy().copy(); rng.shuffle(idx); n = len(idx)
    n_te, n_va = int(round(n * SPLIT_FRAC[2])), int(round(n * SPLIT_FRAC[1]))
    if n < 4: continue
    units.loc[idx[:n_te], "split"] = "test"; units.loc[idx[n_te:n_te + n_va], "split"] = "valid"
trip["split"] = trip.unit.map(units.set_index("unit").split)
for _ in range(20):
    tr = trip[trip.split == "train"]; ok = set(tr["head"]) | set(tr["tail"])
    bad = set(trip.loc[(trip.split != "train") & ~(trip["head"].isin(ok) & trip["tail"].isin(ok)), "unit"])
    if not bad: break
    trip.loc[trip.unit.isin(bad), "split"] = "train"
for s, name in [("train", "train2id.txt"), ("valid", "valid2id.txt"), ("test", "test2id.txt")]:
    d = trip[trip.split == s]
    ref = np.loadtxt(root / "processed/reproduction" / name, skiprows=1, dtype=int, ndmin=2)
    assert sorted(map(tuple, ref.tolist())) == sorted(zip(d["head"], d["tail"], d.relation)), name
print("OK: tái tạo đúng train/valid/test")
''' % (SPLIT_SEED, SPLIT_FRAC, SYMMETRIC)
(SRC / "make_splits.py").write_text(make_splits, encoding="utf-8")
r = subprocess.run([sys.executable, str(SRC / "make_splits.py"), str(MMKG)], capture_output=True, text=True)
print("make_splits.py:", (r.stdout or r.stderr).strip())
assert r.returncode == 0, r.stderr
