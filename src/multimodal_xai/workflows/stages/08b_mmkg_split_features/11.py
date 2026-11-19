HINT = {"P31": "ảnh một [anchor]", "P131": "ảnh công trình/danh thắng ở [anchor]", "P1435": "ảnh nơi được công nhận là [anchor]",
        "P140": "ảnh công trình [anchor]", "P361": "ảnh một công trình thuộc [anchor]", "P825": "ảnh nơi thờ [anchor]",
        "P149": "ảnh công trình theo phong cách [anchor]", "P706": "ảnh nơi nằm trên/cạnh [anchor]", "P84": "ảnh công trình do [anchor] thiết kế",
        "P177": "ảnh cây cầu bắc qua [anchor]", "P708": "ảnh nhà thờ thuộc [anchor]", "NEARBY": "ảnh nơi ở gần [anchor]",
        "CENTURY": "ảnh công trình xây dựng vào [anchor]", "P88": "ảnh công trình do [anchor] cho xây", "P112": "ảnh nơi do [anchor] sáng lập",
        "P138": "ảnh nơi đặt theo tên [anchor]", "P2348": "ảnh công trình thời [anchor]", "P547": "ảnh nơi tưởng niệm [anchor]"}
grp = (Q[Q.query_type == "indirect"].groupby("text_group")
       .agg(splits=("split", lambda s: ",".join(sorted(set(s)))), n_queries=("query_id", "size"),
            direction=("prediction_direction", "first"), relation_index=("relation_index", "first"),
            anchor_entity_index=("anchor_entity_index", "first"), targets=("target_entity_index", lambda s: sorted(set(s)))).reset_index())
grp["relation_vi"] = grp.relation_index.map(RLAB); grp["relation_en"] = grp.relation_index.map(RLAB_EN); grp["pid"] = grp.relation_index.map(PID)
grp["anchor_vi"] = grp.anchor_entity_index.map(LAB); grp["anchor_en"] = grp.anchor_entity_index.map(LAB_EN)
grp["answers_example"] = grp.targets.map(lambda ts: "; ".join(LAB[t] for t in ts[:3]))
grp["hint"] = [HINT.get(p, "ảnh nơi có quan hệ '" + rv + "' với [anchor]").replace("[anchor]", a) + (" (chiều ngược)" if d == "tail" else "")
               for p, rv, a, d in zip(grp.pid, grp.relation_vi, grp.anchor_vi, grp.direction)]
cols = ["text_group", "splits", "n_queries", "pid", "relation_vi", "relation_en", "direction", "anchor_vi", "anchor_en", "answers_example", "hint",
        "text_vi_1", "text_vi_2", "text_en", "written_by", "reviewed_by"]
for c in ["text_vi_1", "text_vi_2", "text_en", "written_by", "reviewed_by"]:
    grp[c] = ""
p_write = REVIEW / "queries_to_write.csv"
if p_write.exists():                                    # đã có người bắt đầu viết: không ghi đè, ghi bản mới bên cạnh
    old = pd.read_csv(p_write, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    if old[["text_vi_1", "text_vi_2", "text_en"]].ne("").any().any():
        p_write = REVIEW / "queries_to_write_new.csv"
        print("queries_to_write.csv đã có câu viết dở: giữ nguyên, ghi khung mới vào", p_write.name)
grp.sort_values(["pid", "anchor_vi"])[cols].to_csv(p_write, index=False, encoding="utf-8-sig")
print(f"Đã ghi {p_write.name}: {len(grp)} nhóm cần viết câu")
