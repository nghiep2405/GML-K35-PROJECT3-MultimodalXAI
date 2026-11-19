ENTS = set(img_q) | set(keep_tail)
triples = [(h, p, t, s, sid) for h, p, t, s, sid in raw_tr if t in ENTS and h in ENTS and h != t]
# quần thể/địa hình không ảnh (type 'site') cũng có tỉnh và quan hệ tới thực thể đã có (không thêm thực thể mới)
sites = [t for t, ty in keep_tail.items() if ty == "site"]
for new_q, provs in province_of(sites).items():
    for new in sorted(provs):
        if NEW_QID[new] in ENTS:
            triples.append((new_q, "P131", NEW_QID[new], "wikidata_P131_chain+NQ202/2025", ""))
for q in sites:
    for pid in PLACE_RELS:
        for v, sid in truthy(ENT[q], pid):
            if v in ENTS and v != q:
                triples.append((q, pid, v, "wikidata", sid))
# nhân vật → dòng họ/triều đại
for t, ty in keep_tail.items():
    if ty == "person":
        for v, sid in truthy(ENT[t], "P53"):
            if v in ENTS and v != t:
                triples.append((t, "P53", v, "wikidata", sid))
# khái niệm → khái niệm
for t, ty in keep_tail.items():
    if ty in ("concept", "period"):
        for v, sid in truthy(ENT.get(t, {}), "P279"):            # thực thể tự sinh (thế kỷ) không có trong ENT
            if v in ENTS and v != t:
                triples.append((t, "P279", v, "wikidata", sid))
# tỉnh mới giáp ranh
prov_in = {t for t, ty in keep_tail.items() if ty == "province"}
NEW_OF_QID = {v: k for k, v in NEW_QID.items()}
border = set()
for p_old, new_a in PROV_NEW.items():
    e = ENT.get(p_old)
    if not e:
        continue
    for v, _ in truthy(e, "P47"):
        new_b = PROV_NEW.get(v)
        if new_b and new_b != new_a:
            border.add((new_a, new_b))
for a, b in sorted(border):
    qa_, qb_ = NEW_QID[a], NEW_QID[b]
    if qa_ in prov_in and qb_ in prov_in:
        triples.append((qa_, "P47", qb_, "wikidata_P47+NQ202/2025", ""))
# ở gần: mỗi thực thể có ảnh nối với NEARBY_K thực thể có ảnh gần nhất trong NEARBY_KM (tọa độ P625), lưu cả hai chiều
def coord(q):
    for v, _ in truthy(ENT[q], "P625"):
        if isinstance(v, dict) and "latitude" in v:
            return float(v["latitude"]), float(v["longitude"])
    return None

def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    return 2 * 6371 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))

if USE_NEARBY:
    XY = {q: c for q in img_q if (c := coord(q))}
    near = set()
    for a in XY:
        for d, b in sorted((km(XY[a], XY[b]), b) for b in XY if b != a)[:NEARBY_K]:
            if d <= NEARBY_KM:
                near.add(tuple(sorted((a, b))))
    for a, b in sorted(near):
        triples += [(a, "NEARBY", b, f"derived_P625_knn{NEARBY_K}_{NEARBY_KM:g}km", ""), (b, "NEARBY", a, f"derived_P625_knn{NEARBY_K}_{NEARBY_KM:g}km", "")]
tr = pd.DataFrame(triples, columns=["head_qid", "pid", "tail_qid", "source", "statement_id"]).drop_duplicates(["head_qid", "pid", "tail_qid"])
# P361 trực tiếp và P527 ngược có thể trùng nhau: giữ bản 'wikidata'
tr = tr.sort_values("source", key=lambda s: s.ne("wikidata")).drop_duplicates(["head_qid", "pid", "tail_qid"]).sort_index()
# thực thể không ảnh mà không còn triple nào thì bỏ
used = set(tr.head_qid) | set(tr.tail_qid)
keep_tail = {t: ty for t, ty in keep_tail.items() if t in used}
ENTS = set(img_q) | set(keep_tail)
tr = tr[tr.head_qid.isin(ENTS) & tr.tail_qid.isin(ENTS)].reset_index(drop=True)
print(f"Triple: {len(tr)}"); display(tr.pid.value_counts().rename("n").to_frame().T)
