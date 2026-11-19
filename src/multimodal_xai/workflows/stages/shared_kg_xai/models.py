class _Model(nn.Module):
    def __init__(self, ent_tot, rel_tot):
        super().__init__(); self.ent_tot = ent_tot; self.rel_tot = rel_tot; self.pi_const = 3.14159265358979323846

class AdvMixRotatE(_Model):
    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0, img_emb=None, text_emb=None):
        super().__init__(ent_tot, rel_tot)
        self.margin = margin; self.epsilon = epsilon; self.dim_e = dim * 2; self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False)
        self.img_proj = nn.Linear(img_emb.shape[1], self.dim_e)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.text_proj = nn.Linear(text_emb.shape[1], self.dim_e)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)
        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.rel_embedding_range = nn.Parameter(torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False)
        self.margin = nn.Parameter(torch.Tensor([margin])); self.margin.requires_grad = False

    def _calc(self, h, t, r, mode):           # bản gốc AdaMF-MAT (mode 'normal')
        pi = self.pi_const
        re_head, im_head = torch.chunk(h, 2, dim=-1); re_tail, im_tail = torch.chunk(t, 2, dim=-1)
        phase_relation = r / (self.rel_embedding_range.item() / pi)
        re_relation, im_relation = torch.cos(phase_relation), torch.sin(phase_relation)
        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)
        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail
        score = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

class TrainOnlyAdaMF(AdvMixRotatE):           # DB15K (04b1)
    def __init__(self, availability, **kw):
        super().__init__(**kw); self.register_buffer("availability", availability.bool())
    def joint(self, ids):
        values = torch.stack([self.ent_embeddings(ids), self.img_proj(self.img_embeddings(ids)), self.text_proj(self.text_embeddings(ids))], dim=1)
        flags = torch.cat([torch.ones_like(self.availability[ids, :1]), self.availability[ids]], dim=1)
        values = values.masked_fill(~flags.unsqueeze(-1), 0.)
        logits = self.ent_attn(torch.tanh(values)).squeeze(-1).masked_fill(~flags, float("-inf"))
        return (logits.softmax(-1).unsqueeze(-1) * values).sum(1)

class MaskedAdvMixRotatE(AdvMixRotatE):       # FB15K-237 (04b2)
    def __init__(self, availability, **kw):
        super().__init__(**kw); self.register_buffer("availability", availability.bool())
    def _joint(self, ids):
        structural = self.ent_embeddings(ids); visual = self.img_proj(self.img_embeddings(ids)); textual = self.text_proj(self.text_embeddings(ids))
        flags = self.availability[ids]; present = torch.cat([torch.ones_like(flags[..., :1]), flags], dim=-1)
        values = torch.stack([structural, visual, textual], dim=-2).masked_fill(~present.unsqueeze(-1), 0.0)
        logits = self.ent_attn(torch.tanh(values)).squeeze(-1).masked_fill(~present, float("-inf"))
        return (torch.softmax(logits, dim=-1).unsqueeze(-1) * values).sum(dim=-2)

class RetrievalAdaMF(nn.Module):
    def __init__(self, N, R, dim, visual, textual, availability, image_entities):
        super().__init__()
        self.kge = MaskedAdvMixRotatE(availability=availability, ent_tot=N, rel_tot=R, dim=dim, margin=12.0, epsilon=2.0,
                                      img_emb=visual.clone(), text_emb=textual.clone())
        self.query_projection = nn.Linear(512, 2 * dim); self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.register_buffer("image_entities", image_entities.clone())

class Scorer:
    """Một checkpoint của nhóm + các phép tính dùng chung cho xếp hạng và giải thích."""
    def __init__(self, D, seed):
        saved = torch.load(D.ckpt[seed], map_location="cpu", weights_only=False)
        # Lớp Model gốc của AdaMF-MAT có 2 hằng số (zero_const = 0, pi_const = π) — không phải tham số học → bỏ khi nạp
        sd = {k: v for k, v in saved["model"].items() if k.rsplit(".", 1)[-1] not in ("zero_const", "pi_const")}; self.alpha = float(saved["alpha_clip"]); self.epoch = saved.get("epoch"); self.kind = D.kind
        if D.kind == "db":
            N, d2 = sd["ent_embeddings.weight"].shape
            core = TrainOnlyAdaMF(availability=sd["availability"], ent_tot=N, rel_tot=sd["rel_embeddings.weight"].shape[0],
                                  dim=d2 // 2, margin=12., epsilon=2., img_emb=sd["img_embeddings.weight"], text_emb=sd["text_embeddings.weight"])
            core.load_state_dict(sd, strict=True); self.model = core.to(DEVICE).eval(); self.core = self.model; self.scale = 1.0
        else:
            k = lambda n: sd["kge." + n]
            N, d2 = k("ent_embeddings.weight").shape
            m = RetrievalAdaMF(N, k("rel_embeddings.weight").shape[0], d2 // 2, k("img_embeddings.weight"), k("text_embeddings.weight"),
                               k("availability"), sd["image_entities"])
            m.load_state_dict(sd, strict=True); self.model = m.to(DEVICE).eval(); self.core = self.model.kge; self.scale = 1.0 / (d2 // 2)
            D.img_ent = sd["image_entities"].clone()          # bản đồ ảnh → thực thể đúng như lúc xếp hạng
        self.model.requires_grad_(False)
        c = self.core
        with torch.no_grad():
            ids = torch.arange(c.ent_tot, device=DEVICE)
            self.S = c.ent_embeddings.weight.detach()                 # cấu trúc
            self.V = c.img_proj(c.img_embeddings(ids)).detach()       # ảnh (đã chiếu)
            self.T = c.text_proj(c.text_embeddings(ids)).detach()     # văn bản (đã chiếu)
            self.flags = torch.cat([torch.ones_like(c.availability[:, :1]), c.availability], 1).bool()
            self.J = self.fuse(self.S, self.V, self.T, self.flags)
        self.margin = float(c.margin); self.rel_range = float(c.rel_embedding_range); self.dim_e = c.dim_e
        self.Rel = c.rel_embeddings.weight.detach()

    def fuse(self, s, v, t, flags, return_w=False):
        values = torch.stack([s, v, t], dim=-2).masked_fill(~flags.unsqueeze(-1), 0.)
        logits = self.core.ent_attn(torch.tanh(values)).squeeze(-1).masked_fill(~flags, float("-inf"))
        w = logits.softmax(-1); j = (w.unsqueeze(-1) * values).sum(-2)
        return (j, w) if return_w else j

    def dist(self, h, t, r):                 # khoảng cách RotatE, broadcast được
        re_h, im_h = h.chunk(2, -1); re_t, im_t = t.chunk(2, -1)
        ph = r / (self.rel_range / math.pi); cr, sr = torch.cos(ph), torch.sin(ph)
        re = re_h * cr - im_h * sr - re_t; im = re_h * sr + im_h * cr - im_t
        return torch.sqrt(re * re + im * im).sum(-1)

    def triple_score(self, hj, tj, r):
        return (self.margin - self.dist(hj, tj, r)) * self.scale

    def entity_scores(self, q, ent_ids, J=None, qproj=None):
        """Điểm đồ thị của từng thực thể ứng viên cho truy vấn q (Series). J: bảng joint (có thể đã bị thay)."""
        J = self.J if J is None else J
        cand = J[ent_ids]
        if q.prediction_direction == "direct":
            if self.kind == "db":
                return None
            return F.normalize(cand, dim=-1) @ qproj
        a = J[int(q.anchor_entity_index)]; r = self.Rel[int(q.relation_index)]
        return self.triple_score(a, cand, r) if q.prediction_direction == "tail" else self.triple_score(cand, a, r)

    def image_scores(self, q, rows, D, J=None, qproj=None, override=None):
        """Điểm đồ thị của các ảnh (max qua thực thể gắn với ảnh). override: {entity: điểm} để thay điểm một thực thể."""
        ids = D.img_ent[torch.as_tensor(rows)].to(DEVICE); present = ids >= 0
        s = self.entity_scores(q, ids.clamp_min(0), J, qproj)
        if s is None:
            return None, None
        if override:
            for e, val in override.items():
                s = torch.where(ids == e, torch.as_tensor(val, device=DEVICE, dtype=s.dtype), s)
        s = s.masked_fill(~present, float("-inf"))
        best = s.max(-1)
        return best.values, ids.gather(-1, best.indices.unsqueeze(-1)).squeeze(-1)

    def qproj(self, text_feat):
        if self.kind == "db":
            return None
        return F.normalize(self.model.query_projection(text_feat.to(DEVICE)), dim=-1)

def zscore(a):
    return (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True, correction=1).clamp_min(1e-6)

def fused_rank_of(clip_top, graph_top, alpha, target_mask, baseline_rank, kind, direction):
    """Đúng công thức đánh giá của nhóm: hạng tốt nhất của tập ảnh đích sau rerank top-k; ngoài top-k giữ hạng CLIP."""
    if graph_top is None or (kind == "db" and (direction == "direct" or alpha == 1.0)):
        return int(baseline_rank)
    if not bool(target_mask.any()):
        return int(baseline_rank)
    fused = alpha * zscore(clip_top) + (1 - alpha) * zscore(graph_top)
    best = fused.masked_fill(~target_mask, float("-inf")).max()
    return 1 + int((fused > best).sum())
