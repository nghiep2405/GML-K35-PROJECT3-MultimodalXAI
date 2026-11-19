def encode_texts(name, texts):
    """CLIP text: text_projection(text_model(...).pooler_output), chuẩn hóa — y hệt notebook của nhóm."""
    D = DATA[name]
    if os.environ.get("XAI2_FAKE_TEXT"):                       # chỉ dùng khi kiểm thử offline
        g = np.random.default_rng(abs(hash(tuple(texts[:3]))) % (2**32))
        return F.normalize(torch.tensor(g.normal(size=(len(texts), 512)), dtype=torch.float32), dim=1)
    from transformers import CLIPModel, CLIPTokenizerFast
    key = D.encoder
    if key not in _TEXT_CACHE:
        tok = CLIPTokenizerFast.from_pretrained(key[0], revision=key[1])
        enc = CLIPModel.from_pretrained(key[0], revision=key[1]).to(DEVICE).eval().requires_grad_(False)
        _TEXT_CACHE[key] = (tok, enc)
    tok, enc = _TEXT_CACHE[key]
    uniq = list(dict.fromkeys(texts)); out = []
    with torch.no_grad():
        for i in range(0, len(uniq), 256):
            t = tok(uniq[i:i + 256], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
            out.append(F.normalize(enc.text_projection(enc.text_model(**t, return_dict=True).pooler_output).float(), dim=1).cpu())
    z = torch.cat(out); pos = {t: i for i, t in enumerate(uniq)}
    return z[torch.tensor([pos[t] for t in texts])]

class Split:
    """CLIP top-k của một split (dùng chung cho mọi seed)."""
    def __init__(self, name, split, qdf):
        D = DATA[name]; self.q = qdf; pool = D.pool[split]
        self.text = encode_texts(name, qdf.query_text.tolist())
        feats = F.normalize(torch.tensor(np.asarray(D.img_feat[pool]), dtype=torch.float32), dim=1).to(DEVICE)
        k = min(TOPK, len(pool)); rows, scores, base = [], [], []
        bs = 32 if D.kind == "db" else 64                # đúng kích thước lô của nhóm → tích vô hướng trùng tới từng bit
        with torch.no_grad():
            for i in range(0, len(qdf), bs):
                s = self.text[i:i + bs].to(DEVICE) @ feats.T
                for j in range(len(s)):
                    p = D.pos[qdf.query_id.iloc[i + j]]
                    pm = torch.as_tensor(np.isin(pool, p), device=DEVICE)
                    base.append(1 + int((s[j] > s[j][pm].max()).sum()))
                    order = torch.argsort(s[j], descending=True, stable=True)[:k]
                    rows.append(torch.as_tensor(pool[order.cpu().numpy()], dtype=torch.int32)); scores.append(s[j][order].cpu())
        self.rows = torch.stack(rows); self.clip = torch.stack(scores); self.base = torch.tensor(base)
        self.target_mask = torch.stack([torch.as_tensor(np.isin(self.rows[i].numpy(), D.pos[qid])) for i, qid in enumerate(qdf.query_id)])
        del feats

    def ctx(self, i, sc):
        """Ngữ cảnh xếp hạng của truy vấn i dưới checkpoint sc."""
        q = self.q.iloc[i]; rows = self.rows[i].long()
        qp = sc.qproj(self.text[i:i + 1])[0] if sc.kind == "fb" else None
        g, ent = sc.image_scores(q, rows, DATA_BY_KIND[sc.kind], qproj=qp)
        return dict(q=q, i=i, rows=rows, clip=self.clip[i].to(DEVICE), graph=g, ent=ent, qproj=qp,
                    base=int(self.base[i]), tmask=self.target_mask[i].to(DEVICE), alpha=sc.alpha, kind=sc.kind)

def rank_in(ctx, graph=None, mask=None):
    return fused_rank_of(ctx["clip"], ctx["graph"] if graph is None else graph, ctx["alpha"],
                         ctx["tmask"] if mask is None else mask, ctx["base"], ctx["kind"], ctx["q"].prediction_direction)

def image_rank(ctx, pos, graph=None):
    """Hạng của ảnh ở vị trí pos trong top-k sau rerank (dùng cho ảnh được trả về)."""
    m = torch.zeros_like(ctx["tmask"]); m[pos] = True
    if ctx["graph"] is None or (ctx["kind"] == "db" and ctx["q"].prediction_direction == "direct"):
        return pos + 1
    return fused_rank_of(ctx["clip"], ctx["graph"] if graph is None else graph, ctx["alpha"], m, pos + 1, ctx["kind"], ctx["q"].prediction_direction)

def top_order(ctx):
    if ctx["graph"] is None or (ctx["kind"] == "db" and (ctx["q"].prediction_direction == "direct" or ctx["alpha"] == 1.0)):
        return torch.arange(len(ctx["rows"]), device=DEVICE)
    fused = ctx["alpha"] * zscore(ctx["clip"]) + (1 - ctx["alpha"]) * zscore(ctx["graph"])
    return torch.argsort(fused, descending=True, stable=True)
