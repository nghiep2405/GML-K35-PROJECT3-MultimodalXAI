RELS = [  # pid, nhãn vi, nhãn en, áp cho loại nguồn
    ("P31", "là một", "instance of", "place"), ("P131", "nằm ở tỉnh", "located in province", "place"),
    ("P1435", "danh hiệu di sản", "heritage designation", "place"), ("P140", "tôn giáo", "religion", "place"),
    ("P361", "thuộc quần thể", "part of", "place"), ("P825", "thờ / dành cho", "dedicated to", "place"),
    ("P88", "người cho xây", "commissioned by", "place"), ("P112", "người sáng lập", "founded by", "place"),
    ("P138", "đặt theo tên", "named after", "place"), ("P2348", "thời kỳ", "time period", "place"),
    ("P149", "phong cách kiến trúc", "architectural style", "place"), ("P706", "nằm trên địa hình", "located on terrain feature", "place"),
    ("P547", "tưởng niệm", "commemorates", "place"), ("P84", "kiến trúc sư", "architect", "place"),
    ("P177", "bắc qua", "crosses", "place"), ("P708", "thuộc giáo phận", "diocese", "place"),
    ("NEARBY", "ở gần", "near (≤ 5 km)", "derived"), ("CENTURY", "xây dựng vào", "built in century", "derived"),
    ("P53", "dòng họ / triều đại", "family", "person"), ("P47", "giáp ranh", "shares border with", "province"),
    ("P279", "loại con của", "subclass of", "concept")]
PLACE_RELS = [r[0] for r in RELS if r[3] == "place" and r[0] != "P131"]
SYNTH = {}                                     # thực thể tự sinh (thế kỷ): id → (nhãn vi, nhãn en)

def century_of(e):
    for v, sid in truthy(e, "P571"):
        if isinstance(v, dict) and v.get("time", "").startswith("+") and int(v.get("precision", 0)) >= 7:
            y = int(v["time"][1:].split("-")[0])
            if y > 0:
                c = (y - 1) // 100 + 1
                return f"VNCENT:{c:02d}", (f"thế kỷ {c}", f"{c}th century"), sid
    return None, None, ""
GENERIC_CLASS = re.compile(r"^(tourist attraction|tourist destination|building|structure|architectural structure|geographic location|location|landmark|place)$", re.I)

img_q = img_ent.qid.tolist()
PROV_OF = province_of(img_q)
raw_tr = []                                  # (h, pid, t, source, statement_id)
for q in img_q:
    e = ENT[q]
    for pid in PLACE_RELS:
        for v, sid in truthy(e, pid):
            if isinstance(v, str) and re.match(r"^Q\d+$", v):
                raw_tr.append((q, pid, v, "wikidata", sid))
    for v, sid in truthy(e, "P527"):            # "có bộ phận" = chiều ngược của "thuộc quần thể"
        if isinstance(v, str) and re.match(r"^Q\d+$", v):
            raw_tr.append((v, "P361", q, "wikidata_P527_inverse", sid))
    for new in sorted(PROV_OF.get(q, [])):
        raw_tr.append((q, "P131", NEW_QID[new], "wikidata_P131_chain+NQ202/2025", ""))
    if USE_CENTURY:
        cid, labs_, sid = century_of(e)
        if cid:
            SYNTH[cid] = labs_; raw_tr.append((q, "CENTURY", cid, "derived_P571_century", sid))
tails = [t for _, p, t, _, _ in raw_tr if p != "P131" and not t.startswith("VNCENT:")]
tails += [h for h, p, t, src, _ in raw_tr if src == "wikidata_P527_inverse" and h not in img_q]
wd_entities(tails)
wd_entities(sorted({c for t in tails if ENT.get(t) for c in all_values(ENT[t], "P31")}), props="labels")

def tail_type(t):
    if t in PROV_QIDS:
        return "province"
    if t.startswith("VNCENT:"):
        return "period"
    e = ENT.get(t)
    if not e:
        return None
    cls = all_values(e, "P31")
    if "Q5" in cls:
        return "person" if truthy(e, "P570") else None          # chỉ nhân vật lịch sử (đã mất)
    if t in img_q:
        return "image_entity"
    if entity_group(e):
        return "site"
    labs = " ".join(lab(c, "en").lower() for c in cls)
    if re.search(r"deity|god|goddess|legendary|mytholog|bodhisattva|buddha|saint|fictional human", labs):
        return "person"
    if re.search(r"\b(dynasty|period|era|kingdom|empire|historical country|former country)\b", labs):
        return "period"
    if re.search(r"\b(river|stream|canal|distributary)\b", labs):
        return "feature"
    if re.search(r"diocese|archdiocese", labs):
        return "diocese"
    return "concept"

freq = Counter(t for h, p, t, _, _ in raw_tr if t not in img_q)
freq.update(h for h, p, t, _, _ in raw_tr if p == "P361" and h not in img_q)   # đầu của P527 ngược
keep_tail = {}
for t, n in freq.items():
    ty = tail_type(t)
    if ty is None or (ty == "concept" and GENERIC_CLASS.match(lab(t, "en") or "")):
        continue
    if n >= MIN_TAIL_FREQ or ty == "province":
        keep_tail[t] = ty
# giới hạn tổng số thực thể ≤ 150: tỉnh luôn giữ (mỗi di tích cần tỉnh của nó), còn lại bỏ đầu ít xuất hiện nhất trước
room = MAX_ENTITIES - len(img_q)
if len(keep_tail) > room:
    ranked = sorted(keep_tail, key=lambda t: (keep_tail[t] != "province", -freq[t], t))
    keep_tail = {t: keep_tail[t] for t in ranked[:room]}
print(f"Thực thể không ảnh: {len(keep_tail)}", Counter(keep_tail.values()).most_common())
