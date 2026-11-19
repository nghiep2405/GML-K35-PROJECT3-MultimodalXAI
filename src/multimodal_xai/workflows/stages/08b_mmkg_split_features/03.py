def norm_label(x):
    s = unicodedata.normalize("NFD", str(x).strip().lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace(" ", "_")
    return {"dung": "dung", "d": "dung", "1": "dung", "sai": "sai", "s": "sai", "0": "sai",
            "khong_ro": "khong_ro", "khongro": "khong_ro", "?": "khong_ro"}.get(s, "")

items = pd.read_csv(REVIEW / "image_check_items.csv", keep_default_na=False)
raters = {}
for r in ["A", "B"]:
    p = REVIEW / f"image_check_rater{r}.csv"
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False, dtype=str)
        df["label"] = df["label"].map(norm_label)
        if df.label.ne("").sum() > 0:
            raters[r] = df.set_index("image_id").label
missing = {r: int((raters[r].reindex(items.image_id).fillna("") == "").sum()) for r in raters}
print("Người chấm có dữ liệu:", list(raters), "| ảnh chưa chấm:", missing)
if len(raters) < 2 and not ALLOW_ONE_RATER:
    raise SystemExit("DỪNG: cần 2 người chấm (image_check_raterA.csv và image_check_raterB.csv có cột label). "
                     "Nếu chỉ có 1 người, đặt os.environ['MMKG_ONE_RATER']='1' rồi chạy lại (κ sẽ ghi là không có).")
assert all(v == 0 for v in missing.values()), f"Còn ảnh chưa chấm: {missing}"

def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = float(np.mean(np.asarray(a) == np.asarray(b)))
    pe = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

chk = items[["image_id", "link_method"]].copy()
for r, s in raters.items():
    chk[r] = chk.image_id.map(s)
rows = []
for m, g in list(chk.groupby("link_method")) + [("tất cả", chk)]:
    row = {"link_method": m, "n": len(g)}
    for r in raters:
        v = g[r][g[r] != "khong_ro"]
        row[f"acc_{r}"] = float((v == "dung").mean()) if len(v) else np.nan
        row[f"khong_ro_{r}"] = int((g[r] == "khong_ro").sum())
    rows.append(row)
acc = pd.DataFrame(rows)
acc["acc_mean"] = acc[[c for c in acc.columns if c.startswith("acc_")]].mean(axis=1)
kappa = cohen_kappa(chk["A"], chk["B"]) if {"A", "B"} <= set(raters) else None
display(acc.round(3))
print("Cohen κ (A–B, 3 nhãn):", None if kappa is None else round(kappa, 3))
d1 = acc.set_index("link_method").acc_mean.get("P373_d1", np.nan)
drop_d1 = bool(np.isfinite(d1) and d1 < D1_MIN_ACC)
bad = set(chk.loc[(chk[list(raters)] == "sai").any(axis=1), "image_id"])
drop = bad | (set(imgs.loc[imgs.link_method.eq("P373_d1"), "image_id"]) if drop_d1 else set())
print(f"Loại {len(bad)} ảnh bị chấm sai" + (f" và mọi ảnh P373_d1 (độ chính xác {d1:.2f} < {D1_MIN_ACC})" if drop_d1 else ""))
imgs = imgs[~imgs.image_id.isin(drop)].reset_index(drop=True)
bad_lic = imgs[imgs.license.str.contains(r"\b(?:NC|ND)\b|GFDL", case=False, regex=True)]
assert bad_lic.empty, f"Có ảnh giấy phép không tự do (NC/ND/GFDL): {bad_lic.image_id.tolist()[:10]}"
review_result = dict(n_items=len(chk), raters=list(raters), kappa=kappa, drop_depth1=drop_d1, n_dropped=len(drop),
                     dropped_image_ids=sorted(drop), accuracy=acc.to_dict(orient="records"))
(REVIEW / "image_check_results.json").write_text(json.dumps(review_result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
