ALL = key(trip)
ANS = defaultdict(set)                                # (r, anchor, dir) → mọi đáp án trong toàn KG
for h, r, t in ALL:
    ANS[(r, t, "head")].add(h); ANS[(r, h, "tail")].add(t)
qrows, relv, filt = [], [], []
def qid(*parts):
    return "q_" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]

first_alias = lambda s, lab: next((a for a in str(s).split("|") if a and a.strip().lower() != lab.strip().lower()), "")
E = ents.set_index("entity_index")
for sp in ["val", "test"]:
    for e, ids in sorted(IMG_OF[sp].items()):
        q = qid("direct", sp, e)
        qrows.append(dict(query_id=q, split=sp, query_type="direct", prediction_direction="direct", anchor_entity_index=pd.NA,
                          relation_index=pd.NA, target_entity_index=e, text_group=f"direct|{e}",
                          text_vi_1=E.loc[e, "label_vi"], text_vi_2=first_alias(E.loc[e, "aliases_vi"], E.loc[e, "label_vi"]),
                          text_en=E.loc[e, "label_en"], text_source="wikidata_label_alias"))
        relv += [dict(query_id=q, image_id=i, target_entity_index=e, split=sp, relevance=1) for i in ids]
    tri = VA if sp == "val" else TE
    for h, r, t, p in zip(tri["head"], tri.relation, tri["tail"], tri.pid):
        if p in NO_QUERY_REL:
            continue
        # quan hệ đối xứng: (h, r, t) và (t, r, h) đều có trong split, nên chỉ sinh chiều head để không đếm trùng một truy vấn
        for target, anchor, direction in ([(h, t, "head")] if p in SYMMETRIC else [(h, t, "head"), (t, h, "tail")]):
            ids = IMG_OF[sp].get(target)
            if not ids:
                continue
            q = qid("indirect", sp, r, anchor, direction, target)
            qrows.append(dict(query_id=q, split=sp, query_type="indirect", prediction_direction=direction, anchor_entity_index=anchor,
                              relation_index=r, target_entity_index=target, text_group=f"{direction}|{r}|{anchor}",
                              text_vi_1="", text_vi_2="", text_en="", text_source="pending_human"))
            relv += [dict(query_id=q, image_id=i, target_entity_index=target, split=sp, relevance=1) for i in ids]
            for x in sorted(ANS[(r, anchor, direction)] - {target}):          # đáp án đúng khác: loại khỏi xếp hạng
                filt += [dict(query_id=q, image_id=i, entity_index=x, reason="other_true_answer") for i in IMG_OF[sp].get(x, [])]
Q = pd.DataFrame(qrows).drop_duplicates("query_id").reset_index(drop=True)
for c in ["anchor_entity_index", "relation_index", "target_entity_index"]:
    Q[c] = Q[c].astype("Int64")
REL = pd.DataFrame(relv).drop_duplicates(["query_id", "image_id"])
FLT = pd.DataFrame(filt, columns=["query_id", "image_id", "entity_index", "reason"]).drop_duplicates(["query_id", "image_id"])
assert not len(REL.merge(FLT, on=["query_id", "image_id"])), "Một ảnh vừa là đáp án vừa bị lọc"
cnt = Q.groupby(["split", "query_type"]).size().unstack(fill_value=0)
print("Số truy vấn:"); display(cnt)
print("Nhóm câu chữ gián tiếp cần viết:", Q[Q.query_type == "indirect"].text_group.nunique(),
      "| truy vấn gián tiếp theo quan hệ:", Q[Q.query_type == "indirect"].relation_index.map(PID).value_counts().to_dict())
for sp, need in [("val", 40), ("test", 60)]:
    n_ind = int(cnt.loc[sp, "indirect"]) if sp in cnt.index and "indirect" in cnt.columns else 0
    if n_ind < need:
        print(f"Lưu ý: {sp} có {n_ind} truy vấn gián tiếp < mục tiêu {need} của kế hoạch.")

# rò rỉ qua văn bản: tên anchor có xuất hiện trong mô tả / mở bài Wikipedia của đáp án không
def textof(e):
    return f"{E.loc[e, 'desc_vi']} {E.loc[e, 'wiki_intro_vi']}".lower()
ind = Q[Q.query_type == "indirect"].copy()
ind["anchor_in_target_text"] = [str(LAB[a]).lower() in textof(t) for a, t in zip(ind.anchor_entity_index, ind.target_entity_index)]
leak = ind.groupby("split").anchor_in_target_text.mean().round(3).to_dict()
leak_by_rel = ind.assign(pid=ind.relation_index.map(PID)).groupby("pid").anchor_in_target_text.mean().round(2).to_dict()
Q = Q.merge(ind[["query_id", "anchor_in_target_text"]], on="query_id", how="left")
print("Tỉ lệ truy vấn gián tiếp có tên anchor trong văn bản của đáp án:", leak, "| theo quan hệ:", leak_by_rel)
