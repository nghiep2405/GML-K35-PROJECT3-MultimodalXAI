# DATA_CARD.md và LICENSES.md — mọi con số sinh từ file, không gõ tay
nt = trip.pid.value_counts()
derived = trip.source.str.startswith("derived").sum()
lic = imgs.license.value_counts()
prov_merge = ents[ents.province_merge_note.ne("")][["label_vi", "province_merge_note"]]
by_type = ents.type.value_counts().to_dict()
acc_tab = acc.copy()
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(("" if (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))) for v in r) + " |")
    return "\n".join(out)

card = f"""# ViHeritage-MMKG — Data card

MMKG tiếng Việt nhỏ về di sản – danh thắng Việt Nam, tự xây cho Project 03 (Học máy với dữ liệu đồ thị, cấp độ 3).
Ngày chụp dữ liệu: **{collect['snapshot_date']}**. Mọi con số dưới đây do notebook `08b` sinh từ file.

## Quy mô
| | Số lượng |
|---|---|
| Thực thể | {N} ({int(ents_out.has_image.sum())} có ảnh) |
| Loại thực thể | {", ".join(f"{k} {v}" for k, v in by_type.items())} |
| Quan hệ | {R} |
| Triple | {len(trip)} (train {len(TR)}, valid {len(VA)}, test {len(TE)}); trong đó {derived} triple suy ra |
| Ảnh | {len(imgs)} (train {int((imgs.retrieval_split == 'train').sum())}, val {len(POOL['val'])}, test {len(POOL['test'])}) |
| Truy vấn | {", ".join(f"{s}/{t}: {n}" for (s, t), n in Q.groupby(['split', 'query_type']).size().items())} |

## Nguồn
- **Thực thể, quan hệ**: Wikidata (statement truthy, hạng tốt nhất), API `wbgetentities` và SPARQL `query.wikidata.org`. CC0.
- **Văn bản**: nhãn, bí danh, mô tả Wikidata (CC0); 2–3 câu đầu phần mở bài Wikipedia tiếng Việt (CC BY-SA 4.0, ghi `wiki_revid_vi`).
- **Ảnh**: Wikimedia Commons, thumbnail 960 px; mỗi ảnh ghi tác giả, giấy phép, trang file (`LICENSES.md`).
- Nhãn tiếng Việt do nhóm dịch: {int(ents.label_vi_source.eq('team_translation').sum())} khái niệm chỉ có nhãn tiếng Anh trên Wikidata;
  {int(ents.label_vi_source.eq('en_fallback').sum())} thực thể vẫn dùng nhãn tiếng Anh (tên riêng).

## Chọn thực thể
Ứng viên: mục ở Việt Nam có danh hiệu di sản (P1435), hoặc thuộc loại công trình/danh thắng phổ biến (lọc theo nhãn loại, danh sách loại ở
`raw/class_whitelist.csv`), có nhãn tiếng Việt, bài Wikipedia tiếng Việt và ảnh. Xếp theo số sitelink. Tối đa 15 danh thắng thiên nhiên,
6 cầu, 8 bảo tàng. Thực thể không ảnh là đầu kia của quan hệ, xuất hiện ≥ 2 lần (tỉnh luôn giữ). Nhân vật chỉ gồm người đã mất.

## Quan hệ và cách tạo triple
{md_table(rels.assign(n_triples=rels.pid.map(nt).fillna(0).astype(int))[["pid", "label_vi", "label_en", "n_triples"]])}

- `P131`: đi theo chuỗi P131 tới cấp tỉnh rồi đổi sang **34 tỉnh, thành sau sắp xếp từ 1/7/2025** (Nghị quyết 202/2025/QH15). Tỉnh bị
  gộp vẫn dùng tên tỉnh mới; ví dụ: {"; ".join(f"{a} ({b})" for a, b in prov_merge.head(4).values)}.
- `P47`: hai tỉnh mới giáp nhau khi hai tỉnh cũ thuộc chúng giáp nhau.
- `P361`: gồm cả `P527` (có bộ phận) đảo chiều.
- **Suy ra** (cột `source` bắt đầu bằng `derived`): `NEARBY` từ tọa độ P625 (mỗi thực thể có ảnh nối với 2 thực thể có ảnh gần nhất trong
  5 km, hai chiều); `CENTURY` từ năm khởi lập P571. Cả hai suy từ thuộc tính nằm ngoài KG, không dùng thống kê trên triple, nên không làm
  lộ triple test.

## Liên kết ảnh – thực thể
Ảnh lấy từ P18 (`P18`), category Commons P373 (`P373_d0`) và category con (`P373_d1`). Lọc: JPEG/PNG, cạnh ngắn ≥ 400 px, giấy phép tự do,
từ khóa tên file/category, CLIP zero-shot (bản đồ, tài liệu, chân dung, đồ ăn…), khử trùng SHA-256 và cosine CLIP ≥ 0,95.
Kiểm tra tay {len(chk)} ảnh ngẫu nhiên chia tầng (người chấm: {", ".join(raters)}; Cohen κ = {"không có" if kappa is None else f"{kappa:.3f}"}):

{md_table(acc_tab.round(3))}

Đã loại {len(drop)} ảnh ({"kể cả mọi ảnh P373_d1" if drop_d1 else "ảnh bị chấm sai"}).

## Chia dữ liệu và chống rò rỉ
- Triple chia {SPLIT_FRAC[0]:.0%}/{SPLIT_FRAC[1]:.0%}/{SPLIT_FRAC[2]:.0%} theo từng quan hệ, seed {SPLIT_SEED}; cặp đối xứng chung split;
  thực thể của triple val/test có trong train. Mã: `source/MMKG/make_splits.py`.
- Ảnh chia theo thực thể; đặc trưng thị giác chỉ lấy từ ảnh train. Truy vấn val/test sinh từ triple và ảnh của chính split đó.
- Truy vấn gián tiếp đánh giá kiểu lọc: đáp án đúng khác của cùng (quan hệ, anchor) bị loại khỏi danh sách xếp hạng.
- Rò rỉ qua văn bản: tỉ lệ truy vấn gián tiếp có tên anchor trong mô tả/mở bài Wikipedia của đáp án = {leak}. Đây là thông tin hợp lệ lúc
  suy luận nhưng được báo cáo, và đo lại khi che văn bản (E2).

## Hạn chế
- Wikidata về di tích Việt Nam thưa (ít P1435, P825); khoảng {derived / len(trip):.0%} triple là quan hệ suy ra.
- Thiên về nơi nổi tiếng (xếp theo sitelink) và ảnh du lịch trên Commons; ít ảnh nội thất, ảnh cũ.
- Địa giới theo 2025; mô tả Wikipedia có thể còn dùng tên tỉnh cũ.

## Điều kiện sử dụng
Chỉ dùng cho học tập và nghiên cứu. Ảnh giữ giấy phép gốc (xem `LICENSES.md`, cần ghi tác giả khi tái sử dụng; ảnh CC BY-SA phải giữ
cùng giấy phép). Văn bản Wikipedia theo CC BY-SA 4.0; dữ liệu Wikidata CC0. Không chứa dữ liệu cá nhân nhạy cảm; nhân vật chỉ là nhân vật
lịch sử đã mất. Ảnh có thể có người qua đường; ảnh chân dung, nhóm người tạo dáng đã được lọc.
"""
(MMKG / "DATA_CARD.md").write_text(card, encoding="utf-8")
lic_rows = imgs.sort_values("image_id")[["image_id", "file_page", "author", "license", "license_url"]]
lines = [f"# Giấy phép ảnh — ViHeritage-MMKG ({len(imgs)} ảnh, nguồn Wikimedia Commons)", "",
         "Phân bố: " + ", ".join(f"{k}: {v}" for k, v in lic.items()), "",
         "| image_id | Trang file | Tác giả | Giấy phép |", "|---|---|---|---|"]
for r in lic_rows.itertuples():
    au = (r.author or "không ghi").replace("|", "/")[:80]
    lines.append(f"| {r.image_id} | {r.file_page} | {au} | [{r.license}]({r.license_url}) |" if r.license_url else
                 f"| {r.image_id} | {r.file_page} | {au} | {r.license} |")
(MMKG / "LICENSES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

files = ["processed/" + p for p in ["entities.parquet", "queries.parquet", "relevance.parquet", "query_filter.parquet", "candidate_pools.parquet",
         "image_feature_index.parquet", "entity_image.parquet", "images.parquet", "clip_image_features.npy", "textual.pt",
         "clip_feature_metadata.json", "triples_split.tsv", "reproduction/entity2id.txt", "reproduction/relation2id.txt",
         "reproduction/train2id.txt", "reproduction/valid2id.txt", "reproduction/test2id.txt"]] + ["DATA_CARD.md", "LICENSES.md"]
manifest = dict(stage="08b_split_features", created=datetime.now(TZ).isoformat(timespec="seconds"), snapshot_date=collect["snapshot_date"],
                split_seed=SPLIT_SEED, split_frac=SPLIT_FRAC, symmetric=sorted(SYMMETRIC), image_encoder=IMG_MODEL, text_encoder=TXT_MODEL,
                counts=dict(entities=N, relations=R, triples=len(trip), train=len(TR), valid=len(VA), test=len(TE), images=len(imgs),
                            queries={f"{s}/{t}": int(n) for (s, t), n in Q.groupby(["split", "query_type"]).size().items()},
                            text_groups_to_write=len(grp), derived_triples=int(derived)),
                image_check=dict(kappa=kappa, drop_depth1=drop_d1, n_dropped=len(drop)), text_leak_rate=leak, text_leak_by_relation=leak_by_rel,
                artifacts={f: dict(path=f, sha256=sha256f(MMKG / f)) for f in files})
(PROC / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(manifest["counts"], ensure_ascii=False))
print("Xong. Bước tiếp: viết câu ở review/queries_to_write.csv → lưu thành review/queries_text.csv → chạy 09.")
