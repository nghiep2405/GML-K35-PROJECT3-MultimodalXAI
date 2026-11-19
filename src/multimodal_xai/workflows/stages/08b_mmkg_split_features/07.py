imgs["retrieval_split"] = "train"
for e, g in imgs.groupby("entity_index"):
    order = g.assign(p18=g.link_method.ne("P18"), rnd=rng.random(len(g))).sort_values(["p18", "rnd"], ascending=[False, True]).index.tolist()
    n = len(order)
    if n >= 3:
        imgs.loc[order[0], "retrieval_split"] = "test"; imgs.loc[order[1], "retrieval_split"] = "val"
    elif n == 2:
        imgs.loc[order[0], "retrieval_split"] = "test"
    else:
        imgs.loc[order[0], "retrieval_split"] = "test"
POOL = {s: imgs.loc[imgs.retrieval_split == s, "image_id"].tolist() for s in ["val", "test"]}
IMG_OF = {s: imgs[imgs.retrieval_split == s].groupby("entity_index").image_id.apply(list).to_dict() for s in ["train", "val", "test"]}
print("Ảnh theo split:", imgs.retrieval_split.value_counts().to_dict(),
      "| thực thể có ảnh train:", len(IMG_OF["train"]), "/", int(ents.has_image.sum()))
