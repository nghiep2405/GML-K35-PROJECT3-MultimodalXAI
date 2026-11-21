# Ô X0 của notebook 10b, 10c, 10d (ViHeritage-MMKG), chép nguyên văn từ 10d_transfer_demo.ipynb trên Drive (bản 17/11/2026).
# Dùng để dựng notebook demo cấp 3 (demo/build_demo_notebooks.py). Không sửa tay: sửa ở notebook gốc rồi chép lại.
# X0. Mã dùng chung của 10b, 10c, 10d — giống hệt nhau ở ba notebook (SHARED_HASH được ghi vào kết quả và đối chiếu).
#     Nạp dữ liệu 08b, mã mô hình và checkpoint 09 đã khóa, chấm truy vấn như 09, và hai bộ giải thích:
#       AdaMF → E3 của cấp 2 (học lại nhúng cấu trúc của e* kiểu Kelpie), GAT → che cạnh trên đồ thị truyền tin (chính xác, không xấp xỉ).
import os, sys, re, json, glob, math, time, hashlib, subprocess, unicodedata, base64, io
from pathlib import Path
from datetime import datetime, timezone, timedelta
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
for _pkg, _mod in [("sentence-transformers", "sentence_transformers"), ("torch_geometric", "torch_geometric")]:
    try:
        __import__(_mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from IPython.display import display

def find_project_root():
    env = os.environ.get("MMKG_PROJECT_ROOT")
    if env:
        return Path(env)
    for p in ["/content/drive/MyDrive/MLGs - Project", "/content/drive/MyDrive/*/MLGs - Project",
              "/content/drive/Shareddrives/*/MLGs - Project", "/content/drive/MyDrive/.shortcut-targets-by-id/*/MLGs - Project"]:
        hits = sorted(glob.glob(p))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError("Không thấy thư mục 'MLGs - Project'.")

PROJECT_ROOT = find_project_root()
FINAL = PROJECT_ROOT / "0. final"
MMKG = FINAL / "data" / "MMKG"; PROC = MMKG / "processed"; REVIEW = MMKG / "review"
OUT09 = Path(os.environ.get("MMKG_OUT", FINAL / "results" / "MMKG"))
XOUT = OUT09 / "XAI_L3"; XOUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TZ = timezone(timedelta(hours=7))
torch.backends.cuda.matmul.allow_tf32 = False
SEEDS, MAIN_SEED = [2027, 2028, 2029], 2027
LANG, TEXT_VARIANT = "vi", "text_vi_1"         # câu dùng khi giải thích: cách nói tiếng Việt thứ nhất, α khóa của (mô hình, vi)
N_BOOT = 2000

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write_json(obj, path):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8"); os.replace(tmp, path)

def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

man = json.loads((PROC / "split_manifest.json").read_text(encoding="utf-8"))
for _f, _spec in man["artifacts"].items():
    assert sha(MMKG / _spec["path"]) == _spec["sha256"], f"File đã đổi sau 08b: {_f}"
HASHES = {f: s["sha256"] for f, s in man["artifacts"].items()}

# ---------- dữ liệu (đọc giống 09) ----------
def read_mapping(p):
    with open(p, encoding="utf-8") as f:
        n = int(f.readline()); rows = [x.rstrip("\n").rsplit("\t", 1) for x in f if x.strip()]
    assert len(rows) == n
    return [r[0] for r in rows]

def read_triples(p):
    with open(p) as f:
        n = int(f.readline()); a = np.loadtxt(f, dtype=np.int64, ndmin=2) if n else np.zeros((0, 3), np.int64)
    assert a.shape == (n, 3)
    return a                                                     # head tail relation

REPRO = PROC / "reproduction"
N, R = len(read_mapping(REPRO / "entity2id.txt")), len(read_mapping(REPRO / "relation2id.txt"))
SPL = {s: read_triples(REPRO / f"{s}2id.txt") for s in ["train", "valid", "test"]}
TRAIN = SPL["train"]
ents = pd.read_parquet(PROC / "entities.parquet"); rels = pd.read_csv(PROC / "relations.csv", keep_default_na=False)
LAB = dict(zip(ents.entity_index, ents.entity_label)); TYPE = dict(zip(ents.entity_index, ents.type))
PID = dict(zip(rels.relation_index, rels.pid)); RLAB = dict(zip(rels.relation_index, rels.label_vi))
DERIVED = set(rels.loc[rels.applies_to == "derived", "relation_index"].astype(int))
Q_ALL = pd.read_parquet(PROC / "queries.parquet"); REL_ALL = pd.read_parquet(PROC / "relevance.parquet")
FLT_ALL = pd.read_parquet(PROC / "query_filter.parquet"); pools = pd.read_parquet(PROC / "candidate_pools.parquet")
idx = pd.read_parquet(PROC / "image_feature_index.parquet"); IMGS = pd.read_parquet(PROC / "images.parquet").set_index("image_id")
img_feat = np.load(PROC / "clip_image_features.npy").astype("float32")
textual = torch.load(PROC / "textual.pt", map_location="cpu", weights_only=True).float()
ROW = idx.set_index("image_id").feature_row; ROW2IMG = idx.set_index("feature_row").image_id
IMG_ENT = idx.sort_values("feature_row").entity_index.to_numpy()
POOL = {s: np.sort(pools.loc[pools.split == s, "image_id"].map(ROW).to_numpy(np.int64)) for s in ["val", "test"]}
tr_rows = idx.loc[idx.retrieval_split == "train", "feature_row"].to_numpy()
train_visual = torch.zeros(N, 512); _cnt = torch.zeros(N)
train_visual.index_add_(0, torch.tensor(IMG_ENT[tr_rows]), torch.tensor(img_feat[tr_rows]))
_cnt.index_add_(0, torch.tensor(IMG_ENT[tr_rows]), torch.ones(len(tr_rows)))
train_visual = F.normalize(train_visual, dim=1)
availability = torch.stack([_cnt > 0, torch.ones(N, dtype=torch.bool)], dim=1)
IMG_N = img_feat / np.linalg.norm(img_feat, axis=1, keepdims=True)
TRAIN_KEYS = {tuple(t) for t in TRAIN.tolist()}
HELD_KEYS = {tuple(t) for t in np.concatenate([SPL["valid"], SPL["test"]]).tolist()}
FACTS_OF, PAIR = {}, {}
for _i, (_h, _t, _r) in enumerate(TRAIN.tolist()):
    FACTS_OF.setdefault(_h, []).append(_i); FACTS_OF.setdefault(_t, []).append(_i)
    PAIR.setdefault((min(_h, _t), max(_h, _t)), []).append(_i)
NEIGH = {e: {int(TRAIN[i, 0]) if int(TRAIN[i, 1]) == e else int(TRAIN[i, 1]) for i in fs} for e, fs in FACTS_OF.items()}
DEG = np.bincount(TRAIN[:, :2].ravel(), minlength=N)
REL_HEADS = {int(r): np.unique(TRAIN[TRAIN[:, 2] == r, 0]) for r in np.unique(TRAIN[:, 2])}
REL_TAILS = {int(r): np.unique(TRAIN[TRAIN[:, 2] == r, 1]) for r in np.unique(TRAIN[:, 2])}

def fact_text(fid=None, trip=None):
    h, t, r = (int(x) for x in (TRAIN[fid] if trip is None else trip))
    return f"{LAB[h]} –[{RLAB[r]}]→ {LAB[t]}"

def image_path(row):
    return MMKG / IMGS.loc[ROW2IMG[int(row)], "image_path"]

# ---------- câu truy vấn (như 09 T4) ----------
TXT = pd.read_csv(REVIEW / "queries_text.csv", encoding="utf-8-sig", keep_default_na=False, dtype=str)
TXT = TXT.set_index("text_group")
for _c in ["text_vi_1", "text_vi_2", "text_en"]:
    TXT[_c] = TXT[_c].map(lambda x: unicodedata.normalize("NFC", x).strip())

def qtext(q, variant=TEXT_VARIANT):
    return (getattr(q, variant) or q.text_vi_1).strip() if q.query_type == "direct" else TXT.loc[q.text_group, variant]

def split_queries(sp):
    q = Q_ALL[Q_ALL.split == sp].sort_values("query_id").reset_index(drop=True)
    rel = REL_ALL[REL_ALL.query_id.isin(q.query_id)].assign(row=lambda d: d.image_id.map(ROW))
    flt = FLT_ALL[FLT_ALL.query_id.isin(q.query_id)].assign(row=lambda d: d.image_id.map(ROW))
    pos = rel.groupby("query_id").row.apply(lambda x: np.unique(x.to_numpy(np.int64))).to_dict()
    drop = flt.groupby("query_id").row.apply(lambda x: set(x.to_numpy(np.int64).tolist())).to_dict()
    return q, pos, drop

# ---------- bộ mã hóa văn bản: đúng revision 09 đã ghim ----------
ENC = json.loads((OUT09 / "text_encoder.json").read_text())
_enc = None
def encode(texts):
    global _enc
    if _enc is None:
        from sentence_transformers import SentenceTransformer
        _enc = SentenceTransformer(ENC["name"], revision=ENC["revision"], device=str(DEVICE))
    return _enc.encode(list(texts), batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")

# ---------- mô hình: đúng mã nguồn và checkpoint 09 ----------
MODEL_SOURCE = (OUT09 / "model_source.py").read_text(encoding="utf-8")
GAT_SOURCE = (OUT09 / "gat_source.py").read_text(encoding="utf-8")
exec(MODEL_SOURCE); exec(GAT_SOURCE)
REL_TXT = torch.from_numpy(encode([f"{a} / {b}" for a, b in zip(rels.label_vi, rels.label_en)]))
LOCK = json.loads((OUT09 / "09_locked.json").read_text(encoding="utf-8"))
CFG = {a: {k: v for k, v in c.items() if k != "config"} for a, c in LOCK["cfg"].items()}
PRIMARY = LOCK.get("primary")
_SEL = FINAL / "results" / "DB15K" / "GAT_vs_AdaMF" / "method_selection.json"
if PRIMARY is None and _SEL.exists():           # 07c chạy sau 09: đọc thẳng kết quả chọn của 07c, không cần chạy lại 09
    PRIMARY = json.loads(_SEL.read_text(encoding="utf-8"))["primary"]
ALPHA = {tuple(k.split("|")): float(v) for k, v in LOCK["alpha"].items()}
ARCH_NAME = {"adamf": "AdaMF", "gat": "GAT"}
_xa = os.environ.get("MMKG_XAI_ARCHS", "primary")
XARCHS = ([PRIMARY] if PRIMARY else ["adamf", "gat"]) if _xa == "primary" else _xa.split(",")

def rebase(p):
    """Đường dẫn trong 09_locked.json là tuyệt đối; mở Drive ở máy khác thì ghép lại theo phần sau 'results/MMKG/'."""
    p = Path(p)
    if p.exists():
        return p
    s = str(p); k = s.find("results/MMKG/")
    q = OUT09 / s[k + len("results/MMKG/"):] if k >= 0 else p
    assert q.exists(), f"Không thấy {p}"
    return q

RUNS = {k: dict(v, dir=rebase(v["dir"])) for k, v in LOCK["runs"].items() if v["variant"] == "full"}
TEST_DIR = rebase(LOCK["test_dir"])

def build_model(cfg, arch, graph):
    kw = dict(ent_tot=N, rel_tot=R, dim=cfg["dim"], margin=cfg["margin"], epsilon=cfg["epsilon"], img_emb=train_visual, text_emb=textual)
    if arch == "adamf":
        return TrainOnlyAdaMF(availability=availability, **kw).to(DEVICE)
    return KGEdgeGAT(availability=availability, graph=graph, rel_text=REL_TXT, **kw).to(DEVICE)

def load_run(name):
    v = RUNS[name]; m = build_model(CFG[v["arch"]], v["arch"], TRAIN)
    st = torch.load(Path(v["dir"]) / "best_model.pt", map_location=DEVICE, weights_only=False)
    m.load_state_dict(st["model"]); m.eval().requires_grad_(False)
    return m, json.loads((Path(v["dir"]) / "complete.json").read_text())

@torch.no_grad()
def all_joint(model):
    model.eval()
    return model.joint(torch.arange(N, device=DEVICE))

def zscore(x):
    return (x - x.mean()) / max(float(x.std(ddof=1)), 1e-6) if len(x) > 1 else x * 0

def rank_of_row(score, i):
    return 1 + int((score > score[i]).sum())

def softmax_p(score, i):
    e = np.exp(score - score.max())
    return float(e[i] / e.sum())

class QCtx:
    """Một truy vấn gián tiếp: ứng viên = pool của split trừ ảnh bị lọc (như 09), điểm CLIP của câu, mặt nạ đáp án."""
    def __init__(self, q, sp, pos, drop, zq):
        self.q, self.sp, self.qid = q, sp, q.query_id
        rows = POOL[sp]; d = drop.get(q.query_id, set()) if drop else set()
        self.rows = rows[~np.isin(rows, list(d))] if d else rows
        self.ents = IMG_ENT[self.rows]; self.uents = np.unique(self.ents)
        self.clip = IMG_N[self.rows] @ zq; self.cz = zscore(self.clip)
        self.pos = np.isin(self.rows, pos.get(q.query_id, [])) if pos else np.zeros(len(self.rows), bool)
        self.anchor, self.rel, self.dir = int(q.anchor_entity_index), int(q.relation_index), q.prediction_direction
    def fused(self, g, alpha):
        return self.clip if alpha == 1.0 else alpha * self.cz + (1 - alpha) * zscore(g)

class RunScorer:
    """Một checkpoint 09 + cách chấm điểm đồ thị; với AdaMF có thêm hàm trộn phương thức để học lại nhúng cấu trúc của một thực thể."""
    def __init__(self, name):
        v = RUNS[name]; self.name, self.arch, self.seed = name, v["arch"], v["seed"]
        self.model, self.info = load_run(name); self.epoch = self.info["best_epoch"]; m = self.model
        self.alpha = ALPHA[(name, LANG)]; self.cfg = CFG[self.arch]
        self.margin, self.range = float(m.margin), float(m.ent_embedding_range)
        self.phase = float(m.rel_embedding_range.item()) / m.pi_const
        with torch.no_grad():
            self.J = all_joint(m); self.Rel = m.rel_embeddings.weight.detach().clone()
            if self.arch == "adamf":
                ids = torch.arange(N, device=DEVICE)
                self.Vp, self.Tp = m.img_proj(m.img_embeddings(ids)), m.text_proj(m.text_embeddings(ids))
                self.flags = torch.cat([torch.ones_like(m.availability[:, :1]), m.availability], 1)
    def dist(self, h, t, r):
        """Khoảng cách RotatE, broadcast được (bằng _calc của mô hình ở mode 'normal')."""
        ph = r / self.phase; rr, ri = torch.cos(ph), torch.sin(ph)
        hr, hi = torch.chunk(h, 2, -1); tr_, ti = torch.chunk(t, 2, -1)
        return torch.stack([hr * rr - hi * ri - tr_, hr * ri + hi * rr - ti], 0).norm(dim=0).sum(-1)
    def ent_score(self, ctx, Je, J=None):
        """Điểm (anchor, r, ·) hoặc (·, r, anchor) cho một hay nhiều nhúng Je."""
        a = (self.J if J is None else J)[ctx.anchor]; r = self.Rel[ctx.rel]
        h, t = (a, Je) if ctx.dir == "tail" else (Je, a)
        return self.margin - self.dist(h, t, r)
    @torch.no_grad()
    def graph(self, ctx, J=None, Rel=None):
        J = self.J if J is None else J; Rel = self.Rel if Rel is None else Rel
        e = torch.as_tensor(ctx.uents, device=DEVICE); a = J[ctx.anchor].expand(len(e), -1); r = Rel[ctx.rel].expand(len(e), -1)
        h, t = (a, J[e]) if ctx.dir == "tail" else (J[e], a)
        s = dict(zip(ctx.uents.tolist(), (self.margin - self.dist(h, t, r)).cpu().numpy()))
        return np.array([s[int(x)] for x in ctx.ents])
    # ----- AdaMF: trộn 3 phương thức (y hệt TrainOnlyAdaMF.joint) cho nhúng cấu trúc X của thực thể e -----
    def fuse(self, X, e):
        fl = self.flags[e]
        vals = torch.stack([X, self.Vp[e].expand_as(X), self.Tp[e].expand_as(X)], 1).masked_fill(~fl[None, :, None], 0.)
        lg = self.model.ent_attn(torch.tanh(vals)).squeeze(-1).masked_fill(~fl[None], float("-inf"))
        return (lg.softmax(-1).unsqueeze(-1) * vals).sum(1)
    def rot(self, x, r, sign=1.0):
        ph = sign * r / self.phase; c, s_ = torch.cos(ph), torch.sin(ph); xr, xi = torch.chunk(x, 2, -1)
        return torch.cat([xr * c - xi * s_, xr * s_ + xi * c], -1)
    def mimic(self, e, fids, keeps, overrides, seed):
        """E3 cấp 2 (Kelpie): học lại nhúng cấu trúc của e từ các bộ ba fids, mỗi biến thể giữ một tập con (keeps) và có thể thay đầu kia
        (overrides: {cột: thực thể mới}). Mọi biến thể cùng khởi tạo và cùng mẫu âm (cùng seed) nên chênh lệch chỉ do tập bộ ba.
        Phép quay RotatE giữ độ dài nên ‖e∘r − t‖ = ‖e − t∘r̄‖ và ‖h∘r − e‖ = ‖e − h∘r‖: tính sẵn đích c cho từng bộ ba / mẫu âm,
        mỗi bước chỉ còn ‖e − c‖ (cùng hàm mục tiêu với 10a, nhanh hơn nhiều)."""
        tr = torch.as_tensor(TRAIN[fids], device=DEVICE); is_head = tr[:, 0] == e
        partner = torch.where(is_head, tr[:, 1], tr[:, 0]); Rr = self.Rel[tr[:, 2]]
        sign = torch.where(is_head, -1.0, 1.0).to(Rr.dtype)[:, None]                       # e là đầu → nhân r̄; e là đuôi → nhân r
        g = torch.Generator().manual_seed(seed)
        neg = torch.randint(0, N, (len(fids), MIMIC["n_neg"]), generator=g).to(DEVICE)
        x0 = ((torch.rand(self.J.shape[1], generator=g) * 2 - 1) * self.range).to(DEVICE)
        Cn = self.rot(self.J[neg], (sign * Rr)[:, None, :])                                 # [F, n, d]
        kp_all = torch.as_tensor(np.stack(keeps), device=DEVICE)
        part = partner.repeat(len(kp_all), 1)
        for v, ov in enumerate(overrides):
            for c, x in (ov or {}).items():
                part[v, c] = int(x)
        Cp = self.rot(self.J[part], (sign * Rr)[None])                                      # [V, F, d]
        def d_(x, c):
            xr, xi = torch.chunk(x, 2, -1); cr, ci = torch.chunk(c, 2, -1)
            return torch.sqrt((xr - cr) ** 2 + (xi - ci) ** 2).sum(-1)
        temp, out = self.cfg["adv_temperature"], []
        step = max(1, int(MIMIC["max_elems"] // max(1, Cn.numel())))                       # số biến thể mỗi lô (giới hạn bộ nhớ)
        for a in range(0, len(kp_all), step):
            kw = kp_all[a:a + step].float(); den = kw.sum(1).clamp_min(1.); C = Cp[a:a + step]
            X = x0.repeat(len(kw), 1).clone().requires_grad_(True); opt = torch.optim.Adam([X], lr=MIMIC["lr"])
            with torch.enable_grad():
                for _ in range(MIMIC["steps"]):
                    Jx = self.fuse(X, e)
                    sp = self.margin - d_(Jx[:, None, :], C)                                     # [V, F]
                    sn = self.margin - d_(Jx[:, None, None, :], Cn[None])                        # [V, F, n]
                    ln = -(torch.softmax(sn * temp, -1).detach() * F.logsigmoid(-sn)).sum(-1)
                    loss = (((-F.logsigmoid(sp) + ln) / 2 * kw).sum(1) / den).sum()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                out.append(self.fuse(X, e).detach())
        return torch.cat(out)
    # ----- GAT: nhúng khi đồ thị truyền tin là `graph` (bỏ / thay cạnh), tham số giữ nguyên -----
    @torch.no_grad()
    def J_batch(self, graphs):
        """Như J_with cho nhiều đồ thị một lúc: ghép thành một đồ thị rời (mỗi biến thể một bản sao nút) → một lượt GATConv.
        Chép đúng KGEdgeGAT.encode; B1 của 10b kiểm tra trùng khớp với J_with."""
        m = self.model; V = len(graphs)
        if not hasattr(self, "_h0"):
            ids = torch.arange(N, device=DEVICE)
            vals = torch.stack([m.ent_embeddings(ids), m.img_proj(m.img_embeddings(ids)), m.text_proj(m.text_embeddings(ids))], 1)
            flags = torch.cat([torch.ones_like(m.availability[:, :1]), m.availability], 1).float()
            self._h0 = (vals * flags.unsqueeze(-1)).sum(1) / flags.sum(1, keepdim=True)
            self._ea = m.edge_projection(m.type_text); self._hin = m.in_proj(self._h0)
        src, dst, et = [], [], []
        for v, gr in enumerate(graphs):
            gr = torch.as_tensor(np.asarray(gr), dtype=torch.long, device=DEVICE).reshape(-1, 3); o = v * N
            src += [gr[:, 0] + o, gr[:, 1] + o]; dst += [gr[:, 1] + o, gr[:, 0] + o]; et += [gr[:, 2], gr[:, 2] + m.rel_tot]
        ei = torch.stack([torch.cat(src), torch.cat(dst)]); ea = self._ea[torch.cat(et)]
        h = self._hin.repeat(V, 1)
        h = m.norm1(h + F.elu(m.gat1(h, ei, edge_attr=ea))); h = m.norm2(h + F.elu(m.gat2(h, ei, edge_attr=ea)))
        return (self._h0.repeat(V, 1) + m.out_proj(h)).view(V, N, -1)
    def J_many(self, graphs, chunk=None):
        chunk = chunk or GAT_CHUNK
        for a in range(0, len(graphs), chunk):
            yield from self.J_batch(graphs[a:a + chunk])
    @torch.no_grad()
    def J_with(self, graph):
        m = self.model; g = torch.as_tensor(np.asarray(graph), dtype=torch.long, device=DEVICE).reshape(-1, 3)
        old = (m.edge_index, m.edge_type)
        m.edge_index = torch.stack([torch.cat([g[:, 0], g[:, 1]]), torch.cat([g[:, 1], g[:, 0]])])
        m.edge_type = torch.cat([g[:, 2], g[:, 2] + m.rel_tot])
        try:
            return m.encode()
        finally:
            m.edge_index, m.edge_type = old

# ---------- giải thích ----------
MIMIC = dict(steps=int(os.environ.get("XAI_STEPS", "300")), lr=0.01, n_neg=16, max_facts=256,   # như 10a
             max_elems=int(os.environ.get("XAI_MAX_ELEMS", "4000000")))
M_CAND = 24                      # AdaMF: số bộ ba ứng viên đo riêng (như 10a)
GAT_MAX_F = int(os.environ.get("XAI_GAT_MAXF", "256"))   # GAT: trần số cạnh trong vùng tiếp nhận 2 bước được che từng cái
K_GRID = [1, 2, 3, 5, 8]; COMP_TARGET = 0.80; N_RANDOM = 10
GAT_CHUNK = int(os.environ.get("XAI_GAT_CHUNK", "32"))   # số biến thể đồ thị mỗi lượt GAT

def hub_score(fid, e):
    h, t, _ = (int(x) for x in TRAIN[fid])
    return DEG[t] if h == e else DEG[h] if t == e else max(DEG[h], DEG[t])

def path_of(fid, e, a):
    """Đường đi ≤ 2 bước anchor … e* chứa bộ ba fid (danh sách fid theo thứ tự anchor → e*), hoặc None."""
    h, t, _ = (int(x) for x in TRAIN[fid]); ends = {h, t}
    if ends == {e, a}:
        return [fid]
    if e in ends and len(ends) == 2:
        x = (ends - {e}).pop()
        if x in NEIGH.get(a, ()):
            return [PAIR[(min(a, x), max(a, x))][0], fid]
    if a in ends and len(ends) == 2:
        x = (ends - {a}).pop()
        if x in NEIGH.get(e, ()):
            return [fid, PAIR[(min(x, e), max(x, e))][0]]
    return None

def perturb_triple(fid, keep, rng):
    """Thay đầu không giữ của bộ ba bằng thực thể từng đứng cùng vai trò với quan hệ đó; bộ ba mới không là bộ ba train/val/test."""
    h, t, r = (int(x) for x in TRAIN[fid]); repl_tail = h == keep
    pool = REL_TAILS[r] if repl_tail else REL_HEADS[r]; old = t if repl_tail else h
    for _ in range(50):
        x = int(rng.choice(pool)) if len(pool) > 2 else int(rng.integers(0, N))
        trip = (h, x, r) if repl_tail else (x, t, r)
        if x not in (keep, old) and trip not in TRAIN_KEYS and trip not in HELD_KEYS:
            return trip
    x = int(rng.integers(0, N))
    return (h, x, r) if repl_tail else (x, t, r)

def universe(rs, ctx, e, rng):
    """F: các bộ ba train có thể ảnh hưởng điểm của e*; cand: các bộ ba được đo riêng. n_all: |F| trước khi cắt trần."""
    a = ctx.anchor
    if rs.arch == "adamf":                                     # E3 cấp 2: bộ ba của chính e*
        fids = np.array(FACTS_OF.get(e, []), np.int64); n_all = len(fids)
        if not n_all:
            return fids, fids, 0
        tr = TRAIN[fids]; partner = np.where(tr[:, 0] == e, tr[:, 1], tr[:, 0])
        with torch.no_grad():
            T_ = torch.as_tensor(tr, device=DEVICE)
            fit = (rs.margin - rs.dist(rs.J[T_[:, 0]], rs.J[T_[:, 1]], rs.Rel[T_[:, 2]])).cpu().numpy()
        near = np.array([p in NEIGH.get(a, ()) for p in partner])
        prio = (partner == a) * 4 + near * 2 + (tr[:, 2] == ctx.rel) * 1
        cand = fids[np.lexsort((-fit, -prio))[:M_CAND]]
        if n_all > MIMIC["max_facts"]:
            rest = rng.choice(np.setdiff1d(fids, cand), MIMIC["max_facts"] - len(cand), replace=False)
            fids = np.concatenate([cand, rest])
        return fids, cand, n_all
    # GAT 2 lớp: điểm của e* phụ thuộc J[e*] và J[anchor]; J[v] phụ thuộc mọi cạnh chạm {v} ∪ N(v)
    nodes = {e, a} | NEIGH.get(e, set()) | NEIGH.get(a, set())
    fids = np.array(sorted({f for v in nodes for f in FACTS_OF.get(v, [])}), np.int64); n_all = len(fids)
    if n_all > GAT_MAX_F:
        tr = TRAIN[fids]
        prio = (tr[:, :2] == e).any(1) * 4 + (tr[:, :2] == a).any(1) * 2 + (tr[:, 2] == ctx.rel) * 1 + rng.random(n_all) * 0.5
        fids = np.sort(fids[np.argsort(-prio, kind="stable")[:GAT_MAX_F]])
    return fids, fids, n_all

def keep_node(fid, e, a):
    h, t, _ = (int(x) for x in TRAIN[fid])
    for v in (e, a):
        if v in (h, t):
            return v
    near = NEIGH.get(e, set()) | NEIGH.get(a, set())
    return h if h in near or t not in near else t

def variant_graphs(rs, ctx, e, fids, specs, seed, g0):
    """specs: [(bỏ: tập fid ⊂ F, thay: {fid: bộ ba mới})] → vector điểm đồ thị trên ứng viên cho từng biến thể.
    AdaMF: học lại nhúng của e* rồi áp chênh lệch giữa hai bản học lại vào điểm gốc (như 10a), các thực thể khác giữ nguyên.
    GAT: tính lại toàn bộ nhúng trên đồ thị đã sửa (mọi thực thể có thể đổi điểm)."""
    if rs.arch == "adamf":
        col = {int(f): i for i, f in enumerate(fids)}
        keeps, ovs = [np.ones(len(fids), bool)], [None]
        for rem, rep in specs:
            m = np.ones(len(fids), bool); m[[col[f] for f in rem]] = False; keeps.append(m)
            ovs.append({col[f]: (tr[1] if tr[0] == e else tr[0]) for f, tr in (rep or {}).items()})
        s = rs.ent_score(ctx, rs.mimic(e, fids, keeps, ovs, seed)).cpu().numpy()
        s_orig = float(rs.ent_score(ctx, rs.J[e])); out = []
        for sv in s[1:]:
            g = g0.copy(); g[ctx.ents == e] = s_orig - (float(s[0]) - float(sv)); out.append(g)
        return out
    ids, graphs = np.arange(len(TRAIN)), []
    for rem, rep in specs:
        T_ = TRAIN.copy()
        for f, tr in (rep or {}).items():
            T_[f] = tr
        graphs.append(T_[~np.isin(ids, list(rem))])
    return [rs.graph(ctx, J) for J in rs.J_many(graphs)]

def importance(rs, ctx, e, fids, cand, seed):
    """δ_f = điểm của e* giảm bao nhiêu khi bỏ riêng f (f ∈ cand); d_all = khi bỏ cả cand. Trả về (δ, d_all)."""
    if rs.arch == "adamf":
        col = {int(f): i for i, f in enumerate(fids)}; keeps = [np.ones(len(fids), bool)]
        for f in cand:
            m = np.ones(len(fids), bool); m[col[int(f)]] = False; keeps.append(m)
        m = np.ones(len(fids), bool); m[[col[int(f)] for f in cand]] = False; keeps.append(m)
        s = rs.ent_score(ctx, rs.mimic(e, fids, keeps, [None] * len(keeps), seed)).cpu().numpy()
        return float(s[0]) - s[1:-1], float(s[0]) - float(s[-1])
    s0 = float(rs.ent_score(ctx, rs.J[e])); ids = np.arange(len(TRAIN))
    s = [float(rs.ent_score(ctx, J[e], J)) for J in rs.J_many([TRAIN[ids != f] for f in cand] + [TRAIN[~np.isin(ids, cand)]])]
    return s0 - np.array(s[:-1]), s0 - s[-1]

def measure(ctx, rs, g, i):
    f = ctx.fused(g, rs.alpha); z = zscore(g)
    return dict(p=softmax_p(f, i), rank=rank_of_row(f, i), pg=softmax_p(z, i), rankg=rank_of_row(z, i), gi=float(g[i]))

def nrm(x, p0):
    return (p0 - x) / p0 if p0 > 0 else np.nan

def explain_query(rs, ctx, Ks, k_full, n_random=N_RANDOM, rand_curve=3, only_importance=False):
    """Giải thích ảnh hạng 1 (điểm cuối α·z(CLIP) + (1−α)·z(đồ thị)) rồi đo fidelity. Ks: các K cần đo; đủ biến thể chỉ ở K = k_full."""
    g0 = rs.graph(ctx); f0 = ctx.fused(g0, rs.alpha); i = int(np.argmax(f0)); e = int(ctx.ents[i]); a = ctx.anchor
    seed = stable_seed(ctx.qid, e, rs.name, rs.epoch); rng = np.random.default_rng(seed)
    fids, cand, n_all = universe(rs, ctx, e, rng)
    meta = dict(run=rs.name, arch=rs.arch, seed=rs.seed, query_id=ctx.qid, split=ctx.sp, direction=ctx.dir, relation=RLAB[ctx.rel],
                anchor=a, anchor_label=LAB[a], row=int(ctx.rows[i]), image_id=ROW2IMG[int(ctx.rows[i])], entity=e, entity_label=LAB[e],
                returned_is_relevant=bool(ctx.pos[i]), alpha=rs.alpha, n_facts_e=len(FACTS_OF.get(e, [])), n_universe=n_all,
                n_F=len(fids), n_cand=len(cand), rank_best_pos=(1 + int((f0 > f0[ctx.pos].max()).sum())) if ctx.pos.any() else np.nan)
    if not len(cand):
        return meta, [], None
    delta, d_all = importance(rs, ctx, e, fids, cand, seed); order = np.argsort(-delta, kind="stable")
    meta.update(d_all=d_all, n_positive_delta=int((delta > 0).sum()))
    S_of = {K: [int(cand[j]) for j in order[:K] if delta[j] > 0] for K in Ks}
    S_of = {K: S for K, S in S_of.items() if S}
    top = [dict(fid=int(cand[j]), delta=float(delta[j]), text=fact_text(int(cand[j]))) for j in order[:max(K_GRID)]]
    expl = dict(meta, K=k_full, S=S_of.get(k_full, []), universe=[int(f) for f in fids], top=top)
    if only_importance or not S_of:                              # chọn K* (10a, C6): độ giảm điểm của e* khi bỏ đồng thời top-K / khi bỏ cả cand
        gv = variant_graphs(rs, ctx, e, fids, [(set(S), None) for S in S_of.values()], seed, g0) if S_of else []
        return meta, [dict(K=K, size_S=len(S), dscore=float(g0[i] - g[i]), comp=float(g0[i] - g[i]) / d_all if d_all > 0 else np.nan)
                      for (K, S), g in zip(S_of.items(), gv)], expl
    specs, tags = [], []
    def add(rem, rep, tag):
        specs.append((set(rem), rep)); tags.append(tag)
    Fset = [int(f) for f in fids]; hubs = sorted(Fset, key=lambda f: (-hub_score(f, e), f))
    for K, S in S_of.items():
        full, k = K == k_full, len(S)
        add(S, None, ("minus_top", K, 0)); add(set(Fset) - set(S), None, ("only_top", K, 0))
        if full:
            add([], {f: perturb_triple(f, keep_node(f, e, a), rng) for f in S}, ("perturb_top", K, 0))
            add(hubs[:k], None, ("minus_deg", K, 0))
        for rr in range(n_random if full else rand_curve):
            rc = [int(x) for x in rng.choice(Fset, k, replace=False)]
            add(rc, None, ("minus_rand", K, rr))
            if full:
                add(set(Fset) - set(rc), None, ("only_rand", K, rr))
    base = measure(ctx, rs, g0, i)
    res = {t: measure(ctx, rs, g, i) for t, g in zip(tags, variant_graphs(rs, ctx, e, fids, specs, seed, g0))}
    NA = dict(p=np.nan, rank=np.nan, pg=np.nan, rankg=np.nan, gi=np.nan); p0, pg0, r0, rg0 = base["p"], base["pg"], base["rank"], base["rankg"]
    rows = []
    for K, S in S_of.items():
        g = lambda kind: res.get((kind, K, 0), NA)
        mr = [v for t, v in res.items() if t[0] == "minus_rand" and t[1] == K]; orr = [v for t, v in res.items() if t[0] == "only_rand" and t[1] == K]
        fp, fm = nrm(g("minus_top")["p"], p0), nrm(g("only_top")["p"], p0)
        c1, c2 = min(max(fp, 0.), 1.), min(max(fm, 0.), 1.)
        paths = [path_of(f, e, a) for f in S]
        rows.append(dict(
            K=K, size_S=len(S), sparsity_F=1 - len(S) / max(1, n_all), dscore_minus_top=base["gi"] - g("minus_top")["gi"],
            comp=(base["gi"] - g("minus_top")["gi"]) / d_all if d_all > 0 else np.nan, dscore_rand=float(np.mean([base["gi"] - x["gi"] for x in mr])),
            p0=p0, rank0=r0, pg0=pg0, rankg0=rg0,
            fid_plus=fp, fid_minus=fm, characterization=(2 * c1 * (1 - c2) / (c1 + 1 - c2)) if (c1 + 1 - c2) > 0 else np.nan,
            drank_minus_top=g("minus_top")["rank"] - r0, PN=float(g("minus_top")["rank"] > 1), PS=float(g("only_top")["rank"] == 1),
            fid_perturb=nrm(g("perturb_top")["p"], p0), drank_perturb=g("perturb_top")["rank"] - r0,
            PN_perturb=float(g("perturb_top")["rank"] > 1) if K == k_full else np.nan,
            fid_plus_deg=nrm(g("minus_deg")["p"], p0), drank_deg=g("minus_deg")["rank"] - r0,
            PN_deg=float(g("minus_deg")["rank"] > 1) if K == k_full else np.nan,
            fid_plus_rand=float(np.mean([nrm(x["p"], p0) for x in mr])), drank_rand=float(np.mean([x["rank"] - r0 for x in mr])),
            PN_rand=float(np.mean([x["rank"] > 1 for x in mr])),
            fid_minus_rand=float(np.mean([nrm(x["p"], p0) for x in orr])) if orr else np.nan,
            PS_rand=float(np.mean([x["rank"] == 1 for x in orr])) if orr else np.nan,
            g_fid_plus=nrm(g("minus_top")["pg"], pg0), g_fid_minus=nrm(g("only_top")["pg"], pg0),
            g_fid_plus_rand=float(np.mean([nrm(x["pg"], pg0) for x in mr])), g_drank_minus_top=g("minus_top")["rankg"] - rg0,
            on_path=float(np.mean([p is not None for p in paths])), touches_e=float(np.mean([e in TRAIN[f, :2] for f in S])),
            share_derived=float(np.mean([int(TRAIN[f, 2]) in DERIVED for f in S])),
            heldout_overlap=int(sum(tuple(int(v) for v in TRAIN[f]) in HELD_KEYS for f in S)),
            in_train=int(all(tuple(int(v) for v in TRAIN[f]) in TRAIN_KEYS for f in S)),
            S_fids=" ".join(map(str, S)), S_text=" | ".join(fact_text(f) for f in S)))
    return meta, rows, expl

def explanation_text(expl):
    e, a = expl["entity"], expl["anchor"]; lines = []
    for f in expl["S"]:
        d = next(x["delta"] for x in expl["top"] if x["fid"] == f); p = path_of(f, e, a)
        lines.append(f"{fact_text(f)}  (δ = {d:+.3f}" + (f"; đường đi: {' ⇒ '.join(fact_text(x) for x in p)})" if p and len(p) > 1 else
                                                         "; nối trực tiếp anchor và e*)" if p else "; ngoài đường đi anchor–e*)"))
    return lines

def thumb_b64(row, size=180):
    from PIL import Image
    im = Image.open(image_path(row)).convert("RGB"); im.thumbnail((size, size))
    b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()
SHARED_HASH = "0730184495b314a7"
print(f"N={N} R={R} train={len(TRAIN)} | phương pháp chính: {PRIMARY} | giải thích: {XARCHS} | thiết bị {DEVICE}")
