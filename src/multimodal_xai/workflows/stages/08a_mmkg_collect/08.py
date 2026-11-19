LICENSE_OK = re.compile(r"^(cc0|cc-zero|public domain|pd\b|pd-|cc[ -]by(-sa)?([ -]\d(\.\d)?)?(\b|$))", re.I)
FILE_BLOCK = re.compile(r"(\bmap\b|\bmaps\b|bản đồ|ban do|\bplan\b|\bplans\b|floor ?plan|sơ đồ|so do|diagram|\blogo|emblem|coat of arms|\bflag\b|"
                        r"cờ|stamp|\btem\b|banknote|coin|document|scan|\bpdf\b|\btext\b|book|newspaper|poster|portrait|chân dung|selfie|"
                        r"\bpeople\b|\bpersons?\b|\bmen\b|\bwomen\b|children|wedding|ceremony|festival|lễ hội|drawing|painting|sketch|"
                        r"illustration|engraving|lithograph|\bicon\b|screenshot|\bsvg\b|locator|location map|satellite|aerial view of the city)", re.I)
CAT_BLOCK = re.compile(r"\b(maps of|plans of|floor plans|diagrams of|portraits|people (of|in|at|on)|men (of|in)|women (of|in)|children (of|in)|"
                       r"coats of arms|flags of|logos|stamps of|documents|drawings of|paintings of|selfies|weddings|festivals?|ceremonies)\b", re.I)
SUBCAT_BLOCK = re.compile(r"(people|persons|portrait|visitors|tourists|events|festival|ceremon|maps|plans|drawings|documents|logos|"
                          r"flags|stamps|videos|audio|interior of shops|night club|restaurants|hotels|food)", re.I)

def file_info(titles):
    """imageinfo cho tối đa 50 file/lần; trả dict title → info."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        for js in mw_query(COMMONS_API, {"titles": "|".join(chunk), "prop": "imageinfo",
                                         "iiprop": "url|size|mime|sha1|extmetadata", "iiurlwidth": str(THUMB_WIDTH),
                                         "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|Categories|ObjectName|DateTimeOriginal|UsageTerms",
                                         "iiextmetadatalanguage": "en"}):
            for pg in js.get("query", {}).get("pages", []):
                if pg.get("missing") or not pg.get("imageinfo"):
                    continue
                out[norm_title(pg["title"])] = pg["imageinfo"][0]
    return out

def cat_members(cat, cmtype="file", limit=500):
    res = []
    for js in mw_query(COMMONS_API, {"list": "categorymembers", "cmtitle": "Category:" + cat, "cmtype": cmtype,
                                     "cmlimit": str(min(limit, 500))}):
        res += [norm_title(m["title"]) for m in js.get("query", {}).get("categorymembers", [])]
        if len(res) >= limit:
            break
    return res[:limit]

def meta_filter(title, ii):
    em = ii.get("extmetadata", {})
    lic = em.get("LicenseShortName", {}).get("value", "")
    cats = em.get("Categories", {}).get("value", "")
    mime = ii.get("mime", "")
    w, h = int(ii.get("width", 0)), int(ii.get("height", 0))
    if mime not in ("image/jpeg", "image/png"):
        return "mime"
    if min(w, h) < MIN_SHORT_SIDE:
        return "small"
    if max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
        return "aspect"
    if re.search(r"\b(nc|nd)\b|non-?commercial|no ?deriv|gfdl", lic, re.I) or not LICENSE_OK.search(lic.strip()):
        return "license"                                    # chỉ CC0 / phạm vi công cộng / CC BY / CC BY-SA; loại NC, ND, GFDL
    if FILE_BLOCK.search(title):
        return "keyword_title"
    if any(CAT_BLOCK.search(c) for c in cats.split("|")):       # không xét category ẩn kiểu "Pages with maps" (mọi ảnh có tọa độ)
        return "keyword_category"
    return ""

def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    r = _request("GET", url, None)
    if not r.headers.get("Content-Type", "").startswith("image/"):
        raise RuntimeError("không phải ảnh")
    dest.write_bytes(r.content)
    N_CALLS["download"] += 1
    time.sleep(DL_SLEEP)
    return r.content

cand_rows = []
t0 = time.time()
for k, r in pool.iterrows():
    q = r.qid
    files = [("File:" + norm_title(f), "P18") for f in r.p18]
    if r.p373:
        files += [(t, "P373_d0") for t in cat_members(r.p373)]
    seen = {}
    for t, m in files:
        seen.setdefault(t, m)                      # P18 ưu tiên khi trùng
    titles = list(seen)[:MAX_FILES_INFO]
    info = file_info(titles)
    ok0 = [t for t in titles if t in info and not meta_filter(t, info[t])]
    if r.p373 and len(ok0) < DEPTH1_IF_BELOW:
        subs = [s.split(":", 1)[1] for s in cat_members(r.p373, "subcat", 100) if not SUBCAT_BLOCK.search(s)][:MAX_SUBCATS]
        extra = []
        for s in subs:
            extra += [t for t in cat_members(s, "file", 100) if t not in seen]
        for t in extra:
            seen.setdefault(t, "P373_d1")
        extra = list(dict.fromkeys(extra))[:MAX_FILES_INFO]
        info.update(file_info(extra))
        titles += extra
    for t in titles:
        ii = info.get(t)
        reason = "missing" if ii is None else meta_filter(t, ii)
        em = (ii or {}).get("extmetadata", {})
        cand_rows.append({"qid": q, "file_title": t, "link_method": seen[t], "reason": reason,
                          "mime": (ii or {}).get("mime", ""), "width": (ii or {}).get("width", 0), "height": (ii or {}).get("height", 0),
                          "sha1_commons": (ii or {}).get("sha1", ""), "thumb_url": (ii or {}).get("thumburl") or (ii or {}).get("url", ""),
                          "file_page": (ii or {}).get("descriptionurl", ""),
                          "license": em.get("LicenseShortName", {}).get("value", ""), "license_url": em.get("LicenseUrl", {}).get("value", ""),
                          "author": strip_html(em.get("Artist", {}).get("value", "")) or strip_html(em.get("Credit", {}).get("value", "")),
                          "date_original": strip_html(em.get("DateTimeOriginal", {}).get("value", ""))})
    if k % 20 == 0:
        print(f"  {k + 1}/{len(pool)} ứng viên, {len(cand_rows)} file, {time.time() - t0:.0f}s")
cdf = pd.DataFrame(cand_rows)
print("Lý do loại (trước khi tải):", cdf.reason.replace("", "đạt").value_counts().to_dict())

# chọn file để tải: P18 trước, rồi độ sâu 0 theo độ phân giải, rồi độ sâu 1; khử trùng sha1 của Commons
order = {"P18": 0, "P373_d0": 1, "P373_d1": 2}
okdf = cdf[cdf.reason.eq("")].copy()
okdf["px"] = okdf.width * okdf.height
okdf = okdf.sort_values(["qid", "link_method", "px"], key=lambda s: s.map(order) if s.name == "link_method" else s, ascending=[True, True, False])
okdf = okdf.drop_duplicates(["qid", "sha1_commons"])
sha_ent = okdf.groupby("sha1_commons").qid.nunique()
shared = set(sha_ent[sha_ent > 1].index)          # cùng một file ở 2 thực thể → liên kết mơ hồ, bỏ
cdf.loc[cdf.sha1_commons.isin(shared) & cdf.reason.eq(""), "reason"] = "shared_between_entities"
okdf = okdf[~okdf.sha1_commons.isin(shared)]
todl = okdf.groupby("qid", sort=False).head(MAX_DL_PER_ENTITY)
print(f"Tải {len(todl)} ảnh cho {todl.qid.nunique()} ứng viên …")
paths, fail_run = {}, 0
for i, r in enumerate(todl.itertuples()):
    ext = ".png" if r.mime == "image/png" else ".jpg"
    dest = CAND / f"{r.sha1_commons}{ext}"
    try:
        download(r.thumb_url, dest); paths[(r.qid, r.file_title)] = dest; fail_run = 0
    except Exception as ex:
        cdf.loc[(cdf.qid == r.qid) & (cdf.file_title == r.file_title), "reason"] = "download_failed"
        fail_run += 1
        if fail_run >= 25:
            raise RuntimeError(f"25 ảnh liên tiếp tải lỗi (lỗi cuối: {ex}). Thường do Wikimedia chặn: đặt MMKG_CONTACT rồi chạy lại; "
                               "các phản hồi đã có vẫn nằm trong cache.") from ex
    if i % 200 == 0:
        print(f"  {i}/{len(todl)} ảnh, {time.time() - t0:.0f}s")
todl = todl[[(q, t) in paths for q, t in zip(todl.qid, todl.file_title)]].copy()
todl["local"] = [str(paths[(q, t)]) for q, t in zip(todl.qid, todl.file_title)]
print(f"Đã tải {len(todl)} ảnh.")
save_cache()
n_have = len(tarfile.open(CAND_TAR).getnames()) if CAND_TAR.exists() else -1
if n_have != sum(1 for _ in CAND.iterdir()):
    tmp = LOCAL / "cand.tar"
    with tarfile.open(tmp, "w") as t:
        for p in sorted(CAND.iterdir()):
            t.add(p, arcname=p.name)
    shutil.copyfile(tmp, CAND_TAR)
