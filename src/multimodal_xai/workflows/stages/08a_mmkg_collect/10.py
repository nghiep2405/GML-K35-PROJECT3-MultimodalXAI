POS_PROMPTS = ["a photo of a temple", "a photo of a pagoda", "a photo of a historic building", "a photo of an ancient gate",
               "a photo of a palace", "a photo of a church", "a photo of a tomb or mausoleum", "a photo of a tower",
               "a photo of a monument or statue", "a photo of a museum building", "a photo of the interior of a temple",
               "a photo of a landscape", "a photo of a bay with islands", "a photo of a cave", "a photo of a mountain",
               "a photo of a lake", "a photo of a bridge", "a photo of a citadel wall"]
NEG_PROMPTS = ["a map", "a floor plan or architectural drawing", "a scanned document or a page of text", "a logo or emblem", "a flag",
               "a postage stamp", "a portrait photo of a person", "a group of people posing for a photo", "a close-up photo of food",
               "a screenshot", "a painting or illustration"]

def load_clip():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer("clip-ViT-B-32", device=dev)
    enc_img = lambda ims: m.encode(ims, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    enc_txt = lambda ts: m.encode(ts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return enc_img, enc_txt

ENC_IMG, ENC_TXT = (globals().get("CLIP_STUB") or load_clip)()
def open_rgb(p):
    im = Image.open(p); im.load()
    return im.convert("RGB")

embs, sha256s = [], []
for i in range(0, len(todl), 64):
    batch = todl.local.iloc[i:i + 64].tolist()
    embs.append(ENC_IMG([open_rgb(p) for p in batch]))
    sha256s += [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in batch]
E = np.concatenate(embs) if embs else np.zeros((0, 512), "float32")
todl["sha256"] = sha256s
T = ENC_TXT(POS_PROMPTS + NEG_PROMPTS)
logits = 100.0 * E @ T.T
prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
todl["neg_prob"] = prob[:, len(POS_PROMPTS):].sum(1)
todl["neg_top"] = [NEG_PROMPTS[j] for j in prob[:, len(POS_PROMPTS):].argmax(1)] if len(todl) else []
todl["emb_row"] = np.arange(len(todl))
keep = todl.neg_prob < NEG_PROB_MAX
print(f"CLIP loại {int((~keep).sum())}/{len(todl)} ảnh:", todl[~keep].neg_top.value_counts().to_dict())
for q, t in zip(todl[~keep].qid, todl[~keep].file_title):
    cdf.loc[(cdf.qid == q) & (cdf.file_title == t), "reason"] = "clip_not_photo"
kd = todl[keep].copy()
# trùng SHA-256 hệt (cùng ảnh, khác tên file)
dup_sha = kd.groupby("sha256").qid.nunique()
bad = set(dup_sha[dup_sha > 1].index)
kd = kd[~kd.sha256.isin(bad)].drop_duplicates(["qid", "sha256"])
# gần trùng theo CLIP
Ek = E[kd.emb_row.values]
S = Ek @ Ek.T
qa = kd.qid.values
drop = set()
for i, j in np.argwhere(np.triu(S >= DUP_COS, 1)):
    if qa[i] == qa[j]:
        drop.add(int(j))                              # cùng thực thể: giữ ảnh đứng trước (P18 / phân giải cao)
    else:
        drop.update([int(i), int(j)])                 # khác thực thể: liên kết mơ hồ, bỏ cả hai
dropped_rows = kd.iloc[sorted(drop)]
for q, t in zip(dropped_rows.qid, dropped_rows.file_title):
    cdf.loc[(cdf.qid == q) & (cdf.file_title == t), "reason"] = "near_duplicate"
kd = kd.iloc[[i for i in range(len(kd)) if i not in drop]].copy()

def pick_diverse(rows):
    """rows: chỉ số dòng của một thực thể theo thứ tự ưu tiên. Dòng đầu (P18 nếu còn, không thì phân giải cao nhất),
    rồi lần lượt thêm ảnh có khoảng cách nhỏ nhất tới các ảnh đã chọn là lớn nhất."""
    er = kd.emb_row.values
    sel = [rows[0]]
    while len(sel) < min(MAX_PER_ENTITY, len(rows)):
        rest = [j for j in rows if j not in sel]
        d = [min(1 - float(E[er[j]] @ E[er[s]]) for s in sel) for j in rest]
        sel.append(rest[int(np.argmax(d))])
    return sel

kd = kd.reset_index(drop=True)
keep_rows = []
for q in dict.fromkeys(kd.qid):
    keep_rows += pick_diverse(list(np.flatnonzero(kd.qid.values == q)))
kd = kd.iloc[keep_rows].reset_index(drop=True)
n_img = kd.groupby("qid").size()
print(f"Sau lọc: {len(kd)} ảnh, {len(n_img)} ứng viên còn ảnh; phân bố số ảnh/thực thể:", n_img.value_counts().sort_index().to_dict())
