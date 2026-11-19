# A0b. Hàm gọi API có cache, thử lại khi 429/5xx, tôn trọng Retry-After.
# Không gửi maxlag: notebook chỉ đọc; maxlag của Wikidata tính cả độ trễ của query service (có lúc > 10 phút) nên sẽ từ chối mọi lần gọi.
N_CALLS = Counter()

def _ckey(url, params):
    s = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _request(method, url, params, headers=None, tries=7):
    last = ""
    for attempt in range(tries):
        try:
            r = SESSION.request(method, url, params=params if method == "GET" else None,
                                data=params if method == "POST" else None, headers=headers, timeout=90)
        except requests.RequestException as e:
            time.sleep(min(60, 2 ** attempt)); last = str(e); continue
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("Retry-After", "")
            time.sleep(int(ra) if ra.isdigit() else min(120, 3 * 2 ** attempt)); last = f"HTTP {r.status_code}"; continue
        if r.status_code == 403:
            raise RuntimeError(f"403 từ {url}. Wikimedia có thể chặn User-Agent; đặt MMKG_CONTACT (email/URL) rồi chạy lại.")
        r.raise_for_status()
        return r
    raise RuntimeError(f"Gọi {url} thất bại sau {tries} lần: {last}")

def get_json(url, params, method="GET"):
    params = dict(params)
    k = _ckey(url, params); p = CACHE / f"{k}.json"
    if p.exists() and not REFRESH:
        return json.loads(p.read_text(encoding="utf-8"))
    for attempt in range(6):
        hdr = {"Accept": "application/sparql-results+json"} if url == SPARQL_URL else None
        js = _request(method, url, params, hdr, tries=3 if url == SPARQL_URL else 7).json()   # SPARQL timeout: thử ít lần
        err = js.get("error", {}) if isinstance(js, dict) else {}
        if err.get("code") in ("maxlag", "ratelimited", "readonly", "internal_api_error_DBQueryError"):
            time.sleep(min(60, 5 * 2 ** attempt)); continue
        break
    if isinstance(js, dict) and "error" in js:
        raise RuntimeError(f"Lỗi API {url}: {js['error']}")
    p.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    N_CALLS[url] += 1
    time.sleep(API_SLEEP)
    return js

def sparql(q):
    js = get_json(SPARQL_URL, {"query": q, "format": "json"}, method="POST")
    out = []
    for b in js["results"]["bindings"]:
        out.append({k: v["value"] for k, v in b.items()})
    return out

def qid_of(uri):
    return uri.rsplit("/", 1)[-1]

def mw_query(api, params):
    """action=query có phân trang 'continue', formatversion=2. Trả về list các phản hồi."""
    base = {"action": "query", "format": "json", "formatversion": "2", **params}
    out, cont = [], {}
    while True:
        js = get_json(api, {**base, **cont})
        out.append(js)
        if "continue" not in js:
            return out
        cont = {k: v for k, v in js["continue"].items()}

ENT = {}          # qid → entity JSON (Wikidata)
REDIRECT = {}
FULL = set()      # QID đã nạp đủ thuộc tính (không chỉ nhãn)
FULL_PROPS = "labels|descriptions|aliases|claims|sitelinks"

def wd_entities(qids, props=FULL_PROPS):
    qids = [q for q in qids if re.match(r"^Q\d+$", q)]
    done = FULL if props == FULL_PROPS else set(ENT)
    need = [q for q in dict.fromkeys(qids) if REDIRECT.get(q, q) not in done]
    for i in range(0, len(need), 50):
        chunk = need[i:i + 50]
        js = get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(chunk), "props": props,
                               "languages": "vi|en", "format": "json"})
        for key, e in js.get("entities", {}).items():
            if "missing" in e:
                continue
            if e["id"] not in FULL:
                ENT[e["id"]] = e
            if props == FULL_PROPS:
                FULL.add(e["id"])
            if key != e["id"]:
                REDIRECT[key] = e["id"]
        for r in js.get("redirects", []) if isinstance(js.get("redirects"), list) else []:
            REDIRECT[r.get("from")] = r.get("to")
    return {q: ENT.get(REDIRECT.get(q, q)) for q in qids}

def lab(qid, lang="vi"):
    e = ENT.get(REDIRECT.get(qid, qid))
    if not e:
        return ""
    v = e.get("labels", {}).get(lang, {}).get("value", "")
    return v or (e.get("labels", {}).get("en", {}).get("value", "") if lang == "vi" else "")

def truthy(e, pid):
    """Giá trị 'truthy' như wdt: hạng preferred nếu có, không thì normal; bỏ deprecated. Trả list (value, statement_id)."""
    st = [s for s in e.get("claims", {}).get(pid, []) if s.get("rank") != "deprecated"
          and s.get("mainsnak", {}).get("snaktype") == "value"]
    pref = [s for s in st if s.get("rank") == "preferred"]
    st = pref or st
    out = []
    for s in st:
        dv = s["mainsnak"]["datavalue"]["value"]
        out.append((dv["id"] if isinstance(dv, dict) and "id" in dv else dv, s.get("id", "")))
    return out

def all_values(e, pid):
    """Mọi giá trị (kể cả hạng normal khi có preferred), bỏ deprecated — dùng cho P31 khi phân loại."""
    out = []
    for s in e.get("claims", {}).get(pid, []):
        if s.get("rank") == "deprecated" or s.get("mainsnak", {}).get("snaktype") != "value":
            continue
        dv = s["mainsnak"]["datavalue"]["value"]
        out.append(dv["id"] if isinstance(dv, dict) and "id" in dv else dv)
    return out

def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()

def norm_title(t):
    t = t.replace("_", " ").strip()
    return t[:1].upper() + t[1:] if t else t

def save_cache():
    """Ghi cache HTTP lên Drive, để runtime mới chạy lại không cần gọi mạng."""
    tmp = LOCAL / "http_cache.tar.gz"
    with tarfile.open(tmp, "w:gz") as t:
        for p in sorted(CACHE.iterdir()):
            t.add(p, arcname=p.name)
    shutil.copyfile(tmp, CACHE_TAR)

print("UA:", UA)
