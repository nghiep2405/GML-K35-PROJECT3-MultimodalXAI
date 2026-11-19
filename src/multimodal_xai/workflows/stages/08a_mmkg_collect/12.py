pool["n_img"] = pool.qid.map(n_img).fillna(0).astype(int)
chosen, n_nat, n_kind = [], 0, Counter()
for tier in [3, MIN_PER_ENTITY]:
    for r in pool.itertuples():
        if len(chosen) >= N_IMG_ENT_MAX:
            break
        if r.qid in chosen or r.n_img < tier:
            continue
        if r.group == "natural" and n_nat >= N_NATURAL_MAX:
            continue
        if r.kind and n_kind[r.kind] >= KIND_QUOTA[r.kind]:
            continue
        chosen.append(r.qid); n_nat += r.group == "natural"; n_kind[r.kind] += 1
    if len(chosen) >= N_IMG_ENT_MIN:
        break
img_ent = pool.set_index("qid").loc[chosen].reset_index()
img_ent["rank"] = np.arange(len(img_ent))
kd = kd[kd.qid.isin(chosen)].copy()
kd["sel_order"] = kd.groupby("qid").cumcount()
while len(kd) > TOTAL_IMG_MAX:                       # bớt ảnh cuối của thực thể đang nhiều ảnh nhất (hòa: thực thể hạng thấp hơn)
    cnt = kd.groupby("qid").size()
    rk = img_ent.set_index("qid")["rank"]
    worst = max(cnt.index, key=lambda q: (cnt[q], rk[q]))
    last = kd[kd.qid == worst].sel_order.max()
    kd = kd[~((kd.qid == worst) & (kd.sel_order == last))]
print(f"Thực thể có ảnh: {len(img_ent)} (natural {int((img_ent.group == 'natural').sum())}, "
      + ", ".join(f"{k} {n_kind[k]}" for k in KIND_QUOTA) + f"); ảnh: {len(kd)}")
assert len(img_ent) >= 60, "Quá ít thực thể có ảnh; nới bộ lọc ở A3/A4 hoặc tăng N_CANDIDATES."

# ---- tỉnh theo địa giới từ 1/7/2025 (Nghị quyết 202/2025/QH15) ----
PROVINCE_2025 = {
    "Hà Nội": ["Hà Nội"], "Huế": ["Thừa Thiên Huế"], "Lai Châu": ["Lai Châu"], "Điện Biên": ["Điện Biên"],
    "Sơn La": ["Sơn La"], "Lạng Sơn": ["Lạng Sơn"], "Quảng Ninh": ["Quảng Ninh"], "Thanh Hóa": ["Thanh Hóa"],
    "Nghệ An": ["Nghệ An"], "Hà Tĩnh": ["Hà Tĩnh"], "Cao Bằng": ["Cao Bằng"],
    "Tuyên Quang": ["Tuyên Quang", "Hà Giang"], "Lào Cai": ["Lào Cai", "Yên Bái"], "Thái Nguyên": ["Thái Nguyên", "Bắc Kạn"],
    "Phú Thọ": ["Phú Thọ", "Vĩnh Phúc", "Hòa Bình"], "Bắc Ninh": ["Bắc Ninh", "Bắc Giang"], "Hưng Yên": ["Hưng Yên", "Thái Bình"],
    "Hải Phòng": ["Hải Phòng", "Hải Dương"], "Ninh Bình": ["Ninh Bình", "Hà Nam", "Nam Định"],
    "Quảng Trị": ["Quảng Trị", "Quảng Bình"], "Đà Nẵng": ["Đà Nẵng", "Quảng Nam"], "Quảng Ngãi": ["Quảng Ngãi", "Kon Tum"],
    "Gia Lai": ["Gia Lai", "Bình Định"], "Khánh Hòa": ["Khánh Hòa", "Ninh Thuận"], "Lâm Đồng": ["Lâm Đồng", "Đắk Nông", "Bình Thuận"],
    "Đắk Lắk": ["Đắk Lắk", "Phú Yên"], "Thành phố Hồ Chí Minh": ["Hồ Chí Minh", "Bình Dương", "Bà Rịa – Vũng Tàu"],
    "Đồng Nai": ["Đồng Nai", "Bình Phước"], "Tây Ninh": ["Tây Ninh", "Long An"], "Cần Thơ": ["Cần Thơ", "Sóc Trăng", "Hậu Giang"],
    "Vĩnh Long": ["Vĩnh Long", "Bến Tre", "Trà Vinh"], "Đồng Tháp": ["Đồng Tháp", "Tiền Giang"], "Cà Mau": ["Cà Mau", "Bạc Liêu"],
    "An Giang": ["An Giang", "Kiên Giang"]}
assert len(PROVINCE_2025) == 34 and sum(map(len, PROVINCE_2025.values())) == 63

def pkey(name):
    s = unicodedata.normalize("NFD", name.lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"^(tinh|thanh pho|tp\.?)\s+", "", s.strip())
    return re.sub(r"[^a-z]+", " ", s).strip()

OLD2NEW = {pkey(o): new for new, olds in PROVINCE_2025.items() for o in olds + [new]}
# mọi đơn vị cấp tỉnh từng là P150 của Việt Nam (mọi hạng, gồm tỉnh cũ) → tên tỉnh mới
prov_rows = sparql("""SELECT ?p ?rank ?end WHERE { wd:Q881 p:P150 ?st . ?st ps:P150 ?p ; wikibase:rank ?rank .
  OPTIONAL { ?st pq:P582 ?end } }""")
prov_items = list(dict.fromkeys(qid_of(r["p"]) for r in prov_rows))
CURRENT = {qid_of(r["p"]) for r in prov_rows if "end" not in r and not r.get("rank", "").endswith("DeprecatedRank")}
wd_entities(prov_items)
PROV_NEW = {}
for p in prov_items:
    for name in [lab(p, "vi")] + [a["value"] for a in ENT.get(p, {}).get("aliases", {}).get("vi", [])]:
        if pkey(name) in OLD2NEW:
            PROV_NEW[p] = OLD2NEW[pkey(name)]; break
unmapped = [lab(p, "vi") for p in prov_items if p not in PROV_NEW]
print(f"Đơn vị cấp tỉnh trên Wikidata: {len(prov_items)}, ánh xạ được sang 34 tỉnh mới: {len(PROV_NEW)}; không ánh xạ: {unmapped[:10]}")
# QID đại diện của mỗi tỉnh mới: mục cùng tên (tỉnh giữ tên), không có thì mã tự đặt
NEW_QID = {}
for new in PROVINCE_2025:
    cands = [p for p, n in PROV_NEW.items() if n == new]
    if cands:   # ưu tiên: còn hiệu lực > cùng tên với tỉnh mới > QID nhỏ (tất định)
        NEW_QID[new] = sorted(cands, key=lambda p: (p not in CURRENT, pkey(lab(p, "vi")) != pkey(new), int(p[1:])))[0]
for new in PROVINCE_2025:
    NEW_QID.setdefault(new, "VNPROV:" + pkey(new).replace(" ", "_"))
PROV_QIDS = set(NEW_QID.values())

PROV_CLASS_KW = re.compile(r"province|municipality|tỉnh|thành phố trực thuộc", re.I)

def province_of(qids):
    """Đi ngược chuỗi P131 (SPARQL P131*) tới đơn vị cấp tỉnh, đổi sang tỉnh mới. Nếu tổ tiên không nằm trong P150 của
    Việt Nam (ví dụ Wikidata đã gỡ tỉnh cũ khỏi P150), so tên tổ tiên có loại 'tỉnh/thành phố trực thuộc trung ương'."""
    out, ancs = {}, defaultdict(list)
    for i in range(0, len(qids), 60):
        vals = " ".join(f"wd:{q}" for q in qids[i:i + 60])
        for r in sparql(f"SELECT ?item ?anc WHERE {{ VALUES ?item {{ {vals} }} ?item wdt:P131* ?anc . }}"):
            it, anc = qid_of(r["item"]), qid_of(r["anc"])
            ancs[it].append(anc)
            if anc in PROV_NEW:
                out.setdefault(it, set()).add(PROV_NEW[anc])
    rest = [q for q in qids if q not in out]
    if rest:
        wd_entities(sorted({a for q in rest for a in ancs[q]}))
        wd_entities(sorted({c for q in rest for a in ancs[q] if ENT.get(a) for c in all_values(ENT[a], "P31")}), props="labels")
        for q in rest:
            for a in ancs[q]:
                e = ENT.get(a)
                if e and pkey(lab(a, "vi")) in OLD2NEW and any(PROV_CLASS_KW.search(lab(c, "en") + " " + lab(c, "vi")) for c in all_values(e, "P31")):
                    out.setdefault(q, set()).add(OLD2NEW[pkey(lab(a, "vi"))])
    return out
