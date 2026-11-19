def load_encoders():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mi, mt = SentenceTransformer(IMG_MODEL, device=dev), SentenceTransformer(TXT_MODEL, device=dev)
    ei = lambda ims: mi.encode(ims, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    et = lambda ts: mt.encode(ts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return ei, et

ENC_IMG, ENC_TXT = (globals().get("ENC_STUB") or load_encoders)()
imgs = imgs.sort_values("image_id").reset_index(drop=True)
imgs["feature_row"] = np.arange(len(imgs))
feats = []
for i in range(0, len(imgs), 64):
    ims = []
    for f in imgs.file.iloc[i:i + 64]:
        im = Image.open(MMKG / f); im.load(); ims.append(im.convert("RGB"))
    feats.append(ENC_IMG(ims))
img_feat = np.concatenate(feats).astype("float32")
assert img_feat.shape == (len(imgs), 512) and np.isfinite(img_feat).all()
ent_text = [". ".join(x for x in [l, d, w] if x) for l, d, w in zip(ents.label_vi, ents.desc_vi, ents.wiki_intro_vi)]
txt_feat = ENC_TXT(ent_text).astype("float32")
assert txt_feat.shape == (N, 512) and np.isfinite(txt_feat).all()
print("Đặc trưng ảnh", img_feat.shape, "| văn bản thực thể", txt_feat.shape)
