def entity_score_vec(sc, q, jvec, qproj=None):
    if q.prediction_direction == "direct":
        return None if sc.kind == "db" else (F.normalize(jvec, dim=-1) @ qproj)
    a = sc.J[int(q.anchor_entity_index)]; r = sc.Rel[int(q.relation_index)]
    return sc.triple_score(a, jvec, r) if q.prediction_direction == "tail" else sc.triple_score(jvec, a, r)

def graph_with(ctx, D, sc, pos, e, new_e_score):
    """Điểm đồ thị top-k khi điểm của thực thể e (ở ảnh vị trí pos) được thay — các thực thể khác của ảnh giữ nguyên."""
    row = int(ctx["rows"][pos]); ids = D.img_ent[row]; ids = ids[ids >= 0].to(DEVICE)
    s = entity_score_vec(sc, ctx["q"], sc.J[ids], ctx["qproj"])
    s = torch.where(ids == e, torch.as_tensor(new_e_score, device=DEVICE, dtype=s.dtype), s)
    g = ctx["graph"].clone(); g[pos] = s.max(); return g

def explain_modality(ctx, D, sc, pos):
    e = int(ctx["ent"][pos]); g0 = float(ctx["graph"][pos]); r0 = image_rank(ctx, pos)
    j, w = sc.fuse(sc.S[e], sc.V[e], sc.T[e], sc.flags[e], return_w=True)
    out = dict(entity=e, label=D.ent_label[e], weights=w.tolist(), present=sc.flags[e].tolist(), g=g0, rank=r0, drop={})
    for m in range(3):
        if not bool(sc.flags[e, m]) or int(sc.flags[e].sum()) == 1:
            continue
        fl = sc.flags[e].clone(); fl[m] = False
        s_new = float(entity_score_vec(sc, ctx["q"], sc.fuse(sc.S[e], sc.V[e], sc.T[e], fl), ctx["qproj"]))
        g_new = graph_with(ctx, D, sc, pos, e, s_new)
        out["drop"][m] = dict(delta=g0 - float(g_new[pos]), rank=image_rank(ctx, pos, g_new))
    if ctx["q"].prediction_direction != "direct":
        a = int(ctx["q"].anchor_entity_index); out["anchor_weights"] = sc.fuse(sc.S[a], sc.V[a], sc.T[a], sc.flags[a], return_w=True)[1].tolist()
    return out

def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

def mimic(sc, D, e, fact_ids, keep, seed):
    """Học lại embedding cấu trúc của e cho V biến thể (keep: V×F bool). Trả về joint [V, d]."""
    tr = torch.as_tensor(D.train[fact_ids], device=DEVICE)
    is_head = (tr[:, 0] == e); partner = torch.where(is_head, tr[:, 1], tr[:, 0])
    P = sc.J[partner]; Rr = sc.Rel[tr[:, 2]]
    g = torch.Generator().manual_seed(seed)
    neg = torch.randint(0, D.N, (len(fact_ids), MIMIC["n_neg"]), generator=g).to(DEVICE); Pn = sc.J[neg]
    rng = float(sc.core.ent_embedding_range)
    x0 = ((torch.rand(sc.dim_e, generator=g) * 2 - 1) * rng).to(DEVICE)
    V = keep.shape[0]; X = x0.repeat(V, 1).clone().requires_grad_(True)
    ve, te, fl = sc.V[e].expand(V, -1), sc.T[e].expand(V, -1), sc.flags[e].expand(V, -1)
    kw = keep.float().to(DEVICE); den = kw.sum(1).clamp_min(1.)
    opt = torch.optim.Adam([X], lr=MIMIC["lr"]); ih = is_head[None, :, None]
    with torch.enable_grad():
        for _ in range(MIMIC["steps"]):
            J = sc.fuse(X, ve, te, fl)[:, None, :]                          # [V,1,d]
            h = torch.where(ih, J, P[None]); t = torch.where(ih, P[None], J)
            sp = sc.margin - sc.dist(h, t, Rr[None])                        # [V,F]
            Jn = J[:, :, None, :]; ihn = ih[..., None]
            hn = torch.where(ihn, Jn, Pn[None]); tn = torch.where(ihn, Pn[None], Jn)
            sn = sc.margin - sc.dist(hn, tn, Rr[None, :, None, :])          # [V,F,n]
            ln = -(torch.softmax(sn * 2.0, -1).detach() * F.logsigmoid(-sn)).sum(-1)
            loss = (((-F.logsigmoid(sp) + ln) / 2 * kw).sum(1) / den).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return sc.fuse(X, ve, te, fl).detach()

def path_of(D, anchor, e, fid):
    h, t, r = (int(x) for x in D.train[fid]); x = t if h == e else h
    if e == anchor:                                 # thực thể của ảnh chính là anchor → chỉ là bộ ba lân cận
        return [(x, e, fid)]
    if x == anchor:
        return [(anchor, e, fid)]
    if anchor in D.facts_of and x in D.neighbors.get(anchor, ()):
        for f2 in D.facts_of[anchor]:
            h2, t2, _ = (int(v) for v in D.train[f2])
            if (h2 == x) or (t2 == x):
                return [(anchor, x, f2), (x, e, fid)]
    return [(x, e, fid)]

def candidate_facts(ctx, D, sc, e, rng):
    fids = np.array(D.facts_of.get(e, []), dtype=np.int64)
    if not len(fids):
        return fids, fids
    q = ctx["q"]; anchor = int(q.anchor_entity_index) if q.prediction_direction != "direct" else -1
    tr = D.train[fids]; partner = np.where(tr[:, 0] == e, tr[:, 1], tr[:, 0])
    with torch.no_grad():
        fit = (sc.margin - sc.dist(sc.J[torch.as_tensor(tr[:, 0], device=DEVICE)], sc.J[torch.as_tensor(tr[:, 1], device=DEVICE)],
                                   sc.Rel[torch.as_tensor(tr[:, 2], device=DEVICE)])).cpu().numpy()
    near = np.array([p in D.neighbors.get(anchor, ()) for p in partner])
    rel_q = int(q.relation_index) if q.prediction_direction != "direct" else -1
    prio = (partner == anchor) * 4 + near * 2 + (tr[:, 2] == rel_q) * 1
    order = np.lexsort((-fit, -prio))
    cand = fids[order[:M_CAND_FACTS]]
    if len(fids) > MIMIC["max_facts"]:                      # bậc rất lớn: giữ ứng viên + mẫu ngẫu nhiên cố định
        rest = np.setdiff1d(fids, cand); rest = rng.choice(rest, MIMIC["max_facts"] - len(cand), replace=False)
        fids = np.concatenate([cand, rest])
    return fids, cand

def explain_facts(ctx, D, sc, pos, Ks, n_random=N_RANDOM):
    """E3 cho ảnh ở vị trí pos. Ks: các K cần đo bỏ đồng thời top-K (ví dụ K_GRID khi chọn K, [K*] khi kiểm chứng)."""
    q = ctx["q"]; e = int(ctx["ent"][pos]); seed = stable_seed(q.query_id, e, sc.alpha, sc.epoch)
    rng = np.random.default_rng(seed)
    fids, cand = candidate_facts(ctx, D, sc, e, rng)
    out = dict(entity=e, n_facts=len(D.facts_of.get(e, [])), facts=[], ok=len(cand) > 0)
    if not len(cand):
        return out
    s_orig = float(entity_score_vec(sc, q, sc.J[e], ctx["qproj"]))
    Fn = len(fids); col = {int(f): i for i, f in enumerate(fids)}
    keep1 = torch.ones(len(cand) + 2, Fn, dtype=torch.bool)
    for i, f in enumerate(cand):
        keep1[1 + i, col[int(f)]] = False
    keep1[-1, [col[int(f)] for f in cand]] = False
    J1 = mimic(sc, D, e, fids, keep1, seed)
    s1 = entity_score_vec(sc, q, J1, ctx["qproj"]).cpu().numpy()
    s_full = float(s1[0]); delta = s_full - s1[1:-1]; d_all = s_full - float(s1[-1])
    order = np.argsort(-delta)
    anchor = int(q.anchor_entity_index) if q.prediction_direction != "direct" else -1
    for i in order:
        f = int(cand[i]); h, t, r = (int(x) for x in D.train[f])
        out["facts"].append(dict(fid=f, head=h, tail=t, rel=r, text=f"{D.ent_label[h]} –[{D.rel_label[r]}]→ {D.ent_label[t]}",
                                 delta=float(delta[i]), path=path_of(D, anchor, e, f) if anchor >= 0 else [(h if t == e else t, e, f)]))
    # Bỏ đồng thời top-K, và K bộ ba ngẫu nhiên (trong mọi bộ ba của e) để so sánh
    variants, tags = [np.ones(Fn, bool)], [("full", 0)]
    for K in Ks:
        top = [int(cand[i]) for i in order[:K] if delta[i] > 0]
        if not top:
            continue
        m = np.ones(Fn, bool); m[[col[f] for f in top]] = False; variants.append(m); tags.append(("top", K))
        for rr in range(n_random):
            m = np.ones(Fn, bool); m[rng.choice(Fn, min(len(top), Fn), replace=False)] = False; variants.append(m); tags.append(("rand", K))
    J2 = mimic(sc, D, e, fids, torch.as_tensor(np.stack(variants)), seed)
    s2 = entity_score_vec(sc, q, J2, ctx["qproj"]).cpu().numpy()
    # Δ đo bằng chênh lệch giữa hai bản học lại, rồi áp vào điểm GỐC của mô hình → hạng luôn tính trên mô hình thật
    r_full = image_rank(ctx, pos)
    res = {}
    for (kind, K), sv in zip(tags[1:], s2[1:]):
        d = float(s2[0]) - float(sv)
        rk = image_rank(ctx, pos, graph_with(ctx, D, sc, pos, e, s_orig - d))
        res.setdefault((kind, K), []).append((d, rk - r_full))
    out.update(s_orig=s_orig, s_full=s_full, d_all=d_all, rank_orig=r_full, rank_full=r_full,
               removal={f"{k}@{K}": dict(dscore=float(np.mean([a for a, _ in v])), drank=float(np.mean([b for _, b in v])))
                        for (k, K), v in res.items()})
    return out
