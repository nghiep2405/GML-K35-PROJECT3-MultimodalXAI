def vi_title(q):
    e = ENT.get(q)
    return e.get("sitelinks", {}).get("viwiki", {}).get("title", "") if e else ""

all_q = img_q + sorted(keep_tail, key=lambda t: (keep_tail[t], lab(t)))
wd_entities([q for q in all_q if not q.startswith("VNPROV:")])
titles = [vi_title(q) for q in all_q if vi_title(q)]
extract = {}
for i in range(0, len(titles), 20):
    chunk = titles[i:i + 20]
    for js in mw_query(VIWIKI_API, {"prop": "extracts|info", "exintro": "1", "explaintext": "1", "exlimit": "20",
                                    "titles": "|".join(chunk), "redirects": "1"}):
        qy = js.get("query", {})
        nmap = {n["from"]: n["to"] for n in qy.get("normalized", [])}
        rmap = {n["from"]: n["to"] for n in qy.get("redirects", [])}
        pages = {p["title"]: p for p in qy.get("pages", []) if not p.get("missing")}
        for t in chunk:
            tt = rmap.get(nmap.get(t, t), nmap.get(t, t))
            if tt in pages and pages[tt].get("extract"):
                extract.setdefault(t, pages[tt])

def first_sentences(txt, n=3, max_chars=450):
    txt = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", txt or "")).strip()
    txt = re.sub(r"\s+([.,;:])", r"\1", txt)
    sents = re.split(r"(?<=[.!?])\s+", txt)
    out = ""
    for s in sents[:n]:
        if len(out) + len(s) > max_chars and out:
            break
        out = (out + " " + s).strip()
    return out

def ent_type(q):
    if q in img_q:
        return img_ent.set_index("qid").loc[q, "group"]
    return keep_tail[q]

ent_rows = []
for i, q in enumerate(all_q):
    e = ENT.get(q, {})
    t = vi_title(q); pg = extract.get(t, {})
    new_name = NEW_OF_QID.get(q)
    ent_rows.append({
        "entity_index": i, "qid": q, "type": ent_type(q), "has_image": q in img_q,
        "label_vi": new_name or SYNTH.get(q, ("", ""))[0] or lab(q, "vi"), "label_en": SYNTH.get(q, ("", ""))[1] or lab(q, "en"),
        "label_vi_source": "nq202_2025" if new_name else ("derived" if q in SYNTH else ("wikidata_vi" if "vi" in e.get("labels", {}) else "en_fallback")),
        "aliases_vi": "|".join(a["value"] for a in e.get("aliases", {}).get("vi", [])),
        "aliases_en": "|".join(a["value"] for a in e.get("aliases", {}).get("en", [])),
        "desc_vi": e.get("descriptions", {}).get("vi", {}).get("value", ""),
        "desc_en": e.get("descriptions", {}).get("en", {}).get("value", ""),
        "wiki_title_vi": t, "wiki_url_vi": ("https://vi.wikipedia.org/wiki/" + t.replace(" ", "_")) if t else "",
        "wiki_revid_vi": pg.get("lastrevid", ""), "wiki_intro_vi": first_sentences(pg.get("extract", "")),
        "province_merge_note": ("gộp từ: " + ", ".join(PROVINCE_2025[new_name])) if new_name and len(PROVINCE_2025[new_name]) > 1 else ""})
ents = pd.DataFrame(ent_rows)
miss = ents[ents.label_vi.eq("")]
assert miss.empty, f"Thiếu nhãn tiếng Việt: {miss.qid.tolist()[:10]}"
print(f"Thực thể: {len(ents)} | nhãn vi lấy tạm từ tiếng Anh: {int(ents.label_vi_source.eq('en_fallback').sum())} | có mở bài Wikipedia vi: {int(ents.wiki_intro_vi.ne('').sum())} | có mô tả vi: {int(ents.desc_vi.ne('').sum())}")
