ents.to_csv(PROC / "entities.csv", index=False)
idx = dict(zip(ents.qid, ents.entity_index))
rel_df = pd.DataFrame([{"relation_index": i, "pid": p, "label_vi": vi, "label_en": en, "applies_to": ap} for i, (p, vi, en, ap) in enumerate(RELS)])
rel_df = rel_df[rel_df.pid.isin(set(tr.pid))].reset_index(drop=True)
rel_df["relation_index"] = np.arange(len(rel_df))
rel_df["n_triples"] = rel_df.pid.map(tr.pid.value_counts())
rel_df.to_csv(PROC / "relations.csv", index=False)
ridx = dict(zip(rel_df.pid, rel_df.relation_index))
tr["head"] = tr.head_qid.map(idx); tr["relation"] = tr.pid.map(ridx); tr["tail"] = tr.tail_qid.map(idx)
tr["retrieved_at"] = SNAPSHOT
tr[["head", "relation", "tail", "head_qid", "pid", "tail_qid", "source", "statement_id", "retrieved_at"]].to_csv(PROC / "triples_all.tsv", sep="\t", index=False)

kd = kd.merge(img_ent[["qid", "rank"]], on="qid").sort_values(["rank", "sel_order"]).reset_index(drop=True)
kd["entity_index"] = kd.qid.map(idx)
kd["image_id"] = [f"img{i:04d}" for i in range(len(kd))]
for r in kd.itertuples():
    ext = Path(r.local).suffix
    dst = IMG / f"{r.image_id}{ext}"
    if not dst.exists():
        shutil.copyfile(r.local, dst)
kd["file"] = [f"images/{i}{Path(p).suffix}" for i, p in zip(kd.image_id, kd.local)]
for p in IMG.iterdir():                                  # xóa ảnh thừa của lần chạy trước (nếu cấu hình đã đổi)
    if f"images/{p.name}" not in set(kd.file):
        p.unlink()
img_cols = ["image_id", "entity_index", "qid", "file", "file_title", "file_page", "author", "license", "license_url", "link_method",
            "sha1_commons", "sha256", "width", "height", "neg_prob", "date_original"]
kd[img_cols].to_csv(PROC / "images.csv", index=False)
cdf.to_csv(RAW / "commons_candidates.csv", index=False)
with open(RAW / "wikidata_entities.jsonl", "w", encoding="utf-8") as f:
    for q in sorted(set(ENT)):
        f.write(json.dumps(ENT[q], ensure_ascii=False) + "\n")
with open(RAW / "wiki_extracts.jsonl", "w", encoding="utf-8") as f:
    for t, pg in sorted(extract.items()):
        f.write(json.dumps({"title": t, **pg}, ensure_ascii=False) + "\n")
pd.DataFrame([{"province_2025": k, "merged_from": "|".join(v), "qid": NEW_QID[k]} for k, v in PROVINCE_2025.items()]).to_csv(RAW / "province_2025_map.csv", index=False)

# ---- kiểm quy mô theo đề (mục 5): 50–150 thực thể, 300–1.000 triple, 200–500 ảnh ----
n_e, n_t, n_i = len(ents), len(tr), len(kd)
print(f"Quy mô: {n_e} thực thể ({int(ents.has_image.sum())} có ảnh), {n_t} triple, {n_i} ảnh")
problems = [m for ok, m in [(50 <= n_e <= 150, "thực thể"), (300 <= n_t <= 1000, "triple"), (200 <= n_i <= 500, "ảnh")] if not ok]
if n_t < 500:
    print(f"Lưu ý: {n_t} triple < mục tiêu 500 của kế hoạch (sàn của đề là 300).")
assert not problems, (f"Ngoài quy mô gợi ý của đề: {problems}. Gửi Claude bảng số triple theo quan hệ ở A6 để chỉnh "
                      "(có thể nới N_IMG_ENT_*, NEARBY_K, MIN_TAIL_FREQ).")

# ---- phiếu kiểm tra ảnh ----
rng = np.random.default_rng(SEED)
share = kd.link_method.value_counts(normalize=True)
quota = {m: max(min(15, int((kd.link_method == m).sum())), int(round(N_REVIEW * s))) for m, s in share.items()}
while sum(quota.values()) > N_REVIEW:
    m = max(quota, key=quota.get); quota[m] -= 1
picks = []
for m, n in quota.items():
    ids = kd[kd.link_method == m].image_id.values
    picks += list(rng.choice(ids, size=min(n, len(ids)), replace=False))
rv = kd[kd.image_id.isin(picks)].merge(ents[["entity_index", "label_vi", "label_en", "desc_vi", "wiki_intro_vi"]], on="entity_index")
rv = rv.sample(frac=1, random_state=SEED).reset_index(drop=True)
rv["item"] = np.arange(1, len(rv) + 1)
base_cols = ["item", "image_id", "label_vi", "label_en", "desc_vi", "file_page"]
rv[base_cols + ["link_method"]].to_csv(REVIEW / "image_check_items.csv", index=False)   # có link_method: chỉ dùng khi tính kết quả
for rater in ["A", "B"]:
    sheet = rv[base_cols].copy(); sheet["label"] = ""; sheet["ghi_chu"] = ""
    p = REVIEW / f"image_check_rater{rater}.csv"
    if p.exists():
        print("Giữ nguyên phiếu đã có (không ghi đè):", p.name)
    else:
        sheet.to_csv(p, index=False, encoding="utf-8-sig")

def thumb_b64(path, side=360):
    im = open_rgb(path); im.thumbnail((side, side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    return base64.b64encode(b.getvalue()).decode()

cards = []
for r in rv.itertuples():
    cards.append(f"""<div class="c"><div class="n">#{r.item} · {r.image_id}</div><img src="data:image/jpeg;base64,{thumb_b64(MMKG / r.file)}">
<div class="t">{html.escape(r.label_vi)}</div><div class="d">{html.escape(r.desc_vi or '')}</div>
<div class="w">{html.escape((r.wiki_intro_vi or '')[:220])}</div><a href="{html.escape(r.file_page)}" target="_blank">trang file Commons</a></div>""")
page = f"""<!doctype html><meta charset="utf-8"><title>Kiểm tra liên kết ảnh – thực thể</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.c{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}}.c img{{width:100%;height:240px;object-fit:contain;background:#eee}}
.n{{font-size:12px;color:#666}}.t{{font-weight:600;margin-top:6px}}.d{{font-size:13px;color:#444}}.w{{font-size:12px;color:#666;margin:4px 0}}a{{font-size:12px}}</style>
<h2>Kiểm tra liên kết ảnh – thực thể ({len(rv)} ảnh)</h2>
<p>Với mỗi ảnh, ghi vào cột <b>label</b> của file CSV của bạn: <b>dung</b> (ảnh đúng là thực thể này), <b>sai</b> (nơi khác / không chụp thực thể),
<b>khong_ro</b> (không đủ thông tin). Chấm độc lập, không trao đổi với người chấm còn lại.</p><div class="g">{''.join(cards)}</div>"""
(REVIEW / "image_check.html").write_text(page, encoding="utf-8")

# ---- lưu cache HTTP lên Drive + manifest ----
save_cache()

def sha256f(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest = {"stage": "08a_collect", "snapshot_date": SNAPSHOT, "user_agent": UA, "refresh": REFRESH,
            "counts": {"entities": n_e, "entities_with_image": int(ents.has_image.sum()), "triples": n_t, "images": n_i,
                       "candidates_checked": len(pool), "commons_files_seen": len(cdf), "review_items": len(rv)},
            "config": {k: globals()[k] for k in ["COVERAGE_MIN", "N_CANDIDATES", "N_IMG_ENT_MIN", "N_IMG_ENT_MAX", "N_NATURAL_MAX",
                                                 "MAX_PER_ENTITY", "MIN_PER_ENTITY", "TOTAL_IMG_MAX", "MIN_SHORT_SIDE", "MAX_ASPECT",
                                                 "THUMB_WIDTH", "NEG_PROB_MAX", "DUP_COS", "MIN_TAIL_FREQ", "SEED"]},
            "api_calls_this_run": dict(N_CALLS),
            "files": {str(p.relative_to(MMKG)): sha256f(p) for p in [PROC / "entities.csv", PROC / "relations.csv", PROC / "triples_all.tsv",
                                                                       PROC / "images.csv", RAW / "commons_candidates.csv"]}}
(RAW / "collect_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest["counts"], ensure_ascii=False))
print("Xong. Bước tiếp: 2 người chấm review/image_check_rater{A,B}.csv, rồi chạy 08b.")
