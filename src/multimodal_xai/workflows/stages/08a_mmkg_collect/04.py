STRUCT_KW = r"\b(temple|pagoda|church|cathedral|basilica|mosque|shrine|citadel|fortress|castle|fort|palace|mausoleum|tomb|museum|tower|bridge|gate|monument|memorial|communal house|archaeological site|archaeological|historic site|heritage site|historic district|ancient town|old quarter|imperial city|theatre|theater|opera house|library|lighthouse|stupa|ruins|market hall|vihara|place of worship|religious building|cemetery|prison|statue|sculpture|square|wall|house|residence|villa)\b"
NATURAL_KW = r"\b(bay|cave|grotto|mountain|peak|hill|lake|waterfall|national park|nature reserve|protected area|scenic area|cultural landscape|landscape|geopark|island|archipelago|beach|valley|mountain pass|lagoon|karst|cape|sand dune)\b"
STRUCT_KW_VI = r"(^|\s)(chùa|đền|đình|miếu|nhà thờ|thánh đường|thành cổ|kinh thành|hoàng thành|lăng|lăng mộ|bảo tàng|tháp|cầu|cổng|tượng đài|khu di tích|di tích|phố cổ|cung|điện|nhà hát|thư viện|hải đăng)(\s|$)"
NATURAL_KW_VI = r"(^|\s)(vịnh|hang|động|núi|đỉnh|đồi|hồ|thác|vườn quốc gia|khu bảo tồn|đảo|quần đảo|bãi biển|thung lũng|đèo|đầm phá|mũi)(\s|$)"
BLOCK_KW = r"\b(human|commune|ward|district|province|city|town|village|township|country|administrative|municipality|rural|urban|company|business|school|university|hospital|hotel|guest house|airport|station|road|street|highway|river|dam|power|stadium|film|organization|organisation|sport|football|election|event|battle|war|people|ethnic|language|dynasty|religion|title|award|ship|aircraft|human settlement|neighborhood|neighbourhood|shopping|office|apartment|skyscraper|residential|tower block|high-rise|diocese|archdiocese|eparchy|ecclesiastical|parish|folklore|folk tale|legend|tale|artefact|artifact|military|operation)\b"
BLOCK_KW_VI = r"(^|\s)(xã|phường|huyện|quận|tỉnh|thành phố|thị xã|thị trấn|làng|sông|trường|bệnh viện|khách sạn|sân bay|nhà ga|đường|dân tộc|trận|chiến tranh|hoạt động|giáo phận|tổng giáo phận|giáo xứ|truyện|di vật|nhà cao tầng)(\s|$)"

_CG = {}
def class_blocked(c):
    return bool(re.search(BLOCK_KW, lab(c, "en").lower()) or re.search(BLOCK_KW_VI, lab(c, "vi").lower()))

def class_group(cls_qid):
    if cls_qid in _CG:
        return _CG[cls_qid]
    _CG[cls_qid] = _class_group(cls_qid)
    return _CG[cls_qid]

def _class_group(cls_qid):
    en, vi = lab(cls_qid, "en").lower(), lab(cls_qid, "vi").lower()
    if re.search(BLOCK_KW, en) or re.search(BLOCK_KW_VI, vi):
        return None
    if re.search(NATURAL_KW, en) or re.search(NATURAL_KW_VI, vi):
        return "natural"
    if re.search(STRUCT_KW, en) or re.search(STRUCT_KW_VI, vi):
        return "structure"
    return None

q_heritage = """SELECT DISTINCT ?item WHERE { ?item wdt:P1435 ?h ; wdt:P17 wd:Q881 . }"""
q_classes = """SELECT ?cls (COUNT(DISTINCT ?item) AS ?n) WHERE {
  ?item wdt:P17 wd:Q881 ; wdt:P31 ?cls ; wdt:P18 ?img .
  ?a schema:about ?item ; schema:isPartOf <https://vi.wikipedia.org/> .
} GROUP BY ?cls HAVING (COUNT(DISTINCT ?item) >= 3) ORDER BY DESC(?n) LIMIT 400"""
route_a = [qid_of(r["item"]) for r in sparql(q_heritage)]
try:
    cls_rows = [(qid_of(r["cls"]), int(r["n"])) for r in sparql(q_classes)]
except RuntimeError as e:
    print("Truy vấn thống kê loại bị lỗi/timeout, chỉ dùng đường 1 (P1435):", e); cls_rows = []
wd_entities([c for c, _ in cls_rows], props="labels")
cls_df = pd.DataFrame([{"class_qid": c, "n_items": n, "label_en": lab(c, "en"), "label_vi": lab(c, "vi"), "group": class_group(c)}
                       for c, n in cls_rows])
white = cls_df[cls_df.group.notna()] if len(cls_df) else cls_df
route_c = []
if len(white):
    vals = " ".join(f"wd:{c}" for c in white.class_qid)
    q_items = f"""SELECT DISTINCT ?item WHERE {{ VALUES ?cls {{ {vals} }}
      ?item wdt:P31 ?cls ; wdt:P17 wd:Q881 .
      ?a schema:about ?item ; schema:isPartOf <https://vi.wikipedia.org/> . }}"""
    route_c = [qid_of(r["item"]) for r in sparql(q_items)]
cand_qids = list(dict.fromkeys(route_a + route_c))
print(f"Đường 1 (P1435): {len(route_a)} mục | loại được chọn: {len(white)}/{len(cls_df)} | đường 2: {len(route_c)} mục | hợp: {len(cand_qids)}")
wd_entities(cand_qids)
# phân loại theo P31; nạp nhãn của mọi loại P31 xuất hiện
all_cls = sorted({c for q in cand_qids if ENT.get(q) for c in all_values(ENT[q], "P31")})
wd_entities(all_cls, props="labels")

def entity_group(e, heritage_default=False):
    """structure / natural / None theo nhãn các loại P31. Mục có danh hiệu di sản mà loại không khớp từ khóa
    (ví dụ chỉ ghi 'building') thì tính là công trình, trừ khi có loại bị chặn (xã, làng, sông…)."""
    cls = all_values(e, "P31")
    gs = [class_group(c) for c in cls]
    if "structure" in gs:
        return "structure"
    if "natural" in gs:
        return "natural"
    if heritage_default and truthy(e, "P1435") and not any(class_blocked(c) for c in cls):
        return "structure"
    return None

rows = []
for q in cand_qids:
    e = ENT.get(q)
    if not e or q in REDIRECT:
        continue
    sl = e.get("sitelinks", {})
    rows.append({"qid": q, "label_vi": lab(q, "vi") if "vi" in e.get("labels", {}) else "", "label_en": lab(q, "en"),
                 "viwiki": sl.get("viwiki", {}).get("title", ""), "n_sitelinks": len(sl),
                 "n_statements": sum(len(v) for v in e.get("claims", {}).values()),
                 "p18": [v for v, _ in truthy(e, "P18")], "p373": (truthy(e, "P373") or [("", "")])[0][0],
                 "heritage": len(truthy(e, "P1435")) > 0, "group": entity_group(e, heritage_default=True),
                 "p31": "|".join(all_values(e, "P31"))})
cand = pd.DataFrame(rows)
cand["has_image"] = cand.p18.map(len).gt(0) | cand.p373.ne("")
cand["ok"] = cand.label_vi.ne("") & cand.viwiki.ne("") & cand.has_image & cand.group.notna()
white.to_csv(RAW / "class_whitelist.csv", index=False)
cov = cand[cand.ok]
print(f"Đạt đủ điều kiện: {len(cov)} mục (structure {int((cov.group == 'structure').sum())}, natural {int((cov.group == 'natural').sum())}); "
      f"có danh hiệu di sản: {int(cov.heritage.sum())}")
print("Loại được chọn nhiều mục nhất:", ", ".join(f"{a} ({n})" for a, n in zip(white.label_en.head(12), white.n_items.head(12))))
if len(cov) < COVERAGE_MIN:
    raise SystemExit(f"DỪNG: chỉ {len(cov)} < {COVERAGE_MIN} mục đạt. Gửi bảng trên cho Claude để mở rộng danh sách loại/từ khóa.")
print("→ ĐI TIẾP: độ phủ đủ.")
