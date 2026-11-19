for D in DATA.values():                                   # bộ ba val/test (để kiểm tra giải thích không chạm dữ liệu giữ lại)
    D.heldout = set()
    for fn in ["valid2id.txt", "test2id.txt"]:
        p = (PHASE2 / ("processed/db15k/experimental_clean_v1/reproduction" if D.kind == "db" else "processed/fb15k237_img/reproduction")) / fn
        if p.exists():
            D.heldout |= {tuple(map(int, x)) for x in read_triples(p)}
    D.train_set = {tuple(map(int, x)) for x in D.train}
    D.rel_heads = {int(r): np.unique(D.train[D.train[:, 2] == r, 0]) for r in np.unique(D.train[:, 2])}
    D.rel_tails = {int(r): np.unique(D.train[D.train[:, 2] == r, 1]) for r in np.unique(D.train[:, 2])}
    print(f"{D.name}: {len(D.heldout)} bộ ba val/test để đối chiếu")

def mimic_multi(sc, D, e, fact_ids, keeps, overrides, seed):
    """Như mimic() của cấp 2, thêm: chia lô biến thể và cho phép thay đầu kia của vài bộ ba (biến thể làm nhiễu).
    Mỗi lô gọi lại cùng seed nên mọi biến thể có cùng khởi tạo và cùng mẫu âm như mimic()."""
    tr = torch.as_tensor(D.train[fact_ids], device=DEVICE)
    is_head = (tr[:, 0] == e); partner = torch.where(is_head, tr[:, 1], tr[:, 0]); Rr = sc.Rel[tr[:, 2]]
    out = []
    for a in range(0, len(keeps), VCHUNK):
        kp = torch.as_tensor(np.stack(keeps[a:a + VCHUNK]))
        part = partner.repeat(len(kp), 1)
        for v, ov in enumerate(overrides[a:a + VCHUNK]):
            for c, x in (ov or {}).items():
                part[v, c] = int(x)
        g = torch.Generator().manual_seed(seed)
        neg = torch.randint(0, D.N, (len(fact_ids), MIMIC["n_neg"]), generator=g).to(DEVICE); Pn = sc.J[neg]
        rng_ = float(sc.core.ent_embedding_range)
        x0 = ((torch.rand(sc.dim_e, generator=g) * 2 - 1) * rng_).to(DEVICE)
        V = len(kp); X = x0.repeat(V, 1).clone().requires_grad_(True); P = sc.J[part]          # [V,F,d]
        ve, te, fl = sc.V[e].expand(V, -1), sc.T[e].expand(V, -1), sc.flags[e].expand(V, -1)
        kw = kp.float().to(DEVICE); den = kw.sum(1).clamp_min(1.)
        opt = torch.optim.Adam([X], lr=MIMIC["lr"]); ih = is_head[None, :, None]
        with torch.enable_grad():
            for _ in range(MIMIC["steps"]):
                J = sc.fuse(X, ve, te, fl)[:, None, :]
                h = torch.where(ih, J, P); t = torch.where(ih, P, J)
                sp = sc.margin - sc.dist(h, t, Rr[None])
                Jn = J[:, :, None, :]; ihn = ih[..., None]
                hn = torch.where(ihn, Jn, Pn[None]); tn = torch.where(ihn, Pn[None], Jn)
                sn = sc.margin - sc.dist(hn, tn, Rr[None, :, None, :])
                ln = -(torch.softmax(sn * 2.0, -1).detach() * F.logsigmoid(-sn)).sum(-1)
                loss = (((-F.logsigmoid(sp) + ln) / 2 * kw).sum(1) / den).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out.append(sc.fuse(X, ve, te, fl).detach())
    return torch.cat(out)

def prob_at(ctx, pos, graph):
    f = ctx["alpha"] * zscore(ctx["clip"]) + (1 - ctx["alpha"]) * zscore(graph)
    return float(torch.softmax(f, -1)[pos])

def perturb_partner(D, fid, e, rng):
    """Đầu kia mới: thực thể từng đứng cùng vai trò với quan hệ r trong train (thay cho 'cùng loại'), và (e, r, x') không là bộ ba train."""
    h, t, r = (int(x) for x in D.train[fid])
    pool = D.rel_tails[r] if h == e else D.rel_heads[r]
    for _ in range(20):
        x = int(rng.choice(pool)) if len(pool) else int(rng.integers(0, D.N))
        trip = (e, x, r) if h == e else (x, e, r)
        if x != e and trip not in D.train_set:
            return x
    return int(rng.integers(0, D.N))

def nrm(x, p0):
    return (p0 - x) / p0 if p0 > 0 else np.nan

def explain_quant(ctx, D, sc, pos, Ks, n_random, k_full):
    """Ks: các K cần đo. Đủ mọi biến thể chỉ ở K = k_full (K*); các K khác (đường cong) chỉ đo bỏ / chỉ giữ S và 3 lần bỏ ngẫu nhiên."""
    q = ctx["q"]; e = int(ctx["ent"][pos]); seed = stable_seed(q.query_id, e, sc.alpha, sc.epoch)
    rng = np.random.default_rng(seed)
    fids, cand = candidate_facts(ctx, D, sc, e, rng)
    meta = dict(entity=e, n_facts=len(D.facts_of.get(e, [])), n_F=len(fids), n_cand=len(cand))
    if not len(cand):
        return meta, []
    s_orig = float(entity_score_vec(sc, q, sc.J[e], ctx["qproj"]))
    Fn = len(fids); col = {int(f): i for i, f in enumerate(fids)}
    keep1 = torch.ones(len(cand) + 2, Fn, dtype=torch.bool)                       # lượt 1: y hệt E3 cấp 2
    for i, f in enumerate(cand):
        keep1[1 + i, col[int(f)]] = False
    keep1[-1, [col[int(f)] for f in cand]] = False
    s1 = entity_score_vec(sc, q, mimic(sc, D, e, fids, keep1, seed), ctx["qproj"]).cpu().numpy()
    delta = float(s1[0]) - s1[1:-1]; order = np.argsort(-delta)
    tr = D.train[fids]; partner = np.where(tr[:, 0] == e, tr[:, 1], tr[:, 0])
    deg = np.array([len(D.facts_of.get(int(p), [])) for p in partner])
    anchor = int(q.anchor_entity_index)
    keeps, ovs, tags, S_of = [np.ones(Fn, bool)], [None], [("full", 0, 0)], {}
    def add(m, ov, tag):
        keeps.append(m); ovs.append(ov); tags.append(tag)
    for K in Ks:
        S = [int(cand[i]) for i in order[:K] if delta[i] > 0]
        if not S:
            continue
        S_of[K] = S; cs = [col[f] for f in S]; k = len(S); full = K == k_full
        m = np.ones(Fn, bool); m[cs] = False; add(m, None, ("minus_top", K, 0))
        m = np.zeros(Fn, bool); m[cs] = True; add(m, None, ("only_top", K, 0))
        if full:
            add(np.ones(Fn, bool), {c: perturb_partner(D, fids[c], e, rng) for c in cs}, ("perturb_top", K, 0))
            m = np.ones(Fn, bool); m[np.argsort(-deg, kind="stable")[:k]] = False; add(m, None, ("minus_deg", K, 0))
        for rr in range(n_random if full else min(3, n_random)):
            rc = rng.choice(Fn, k, replace=False)
            m = np.ones(Fn, bool); m[rc] = False; add(m, None, ("minus_rand", K, rr))
            if full:
                m = np.zeros(Fn, bool); m[rc] = True; add(m, None, ("only_rand", K, rr))
    if not S_of:
        return meta, []
    s2 = entity_score_vec(sc, q, mimic_multi(sc, D, e, fids, keeps, ovs, seed), ctx["qproj"]).cpu().numpy()
    p0, r0 = prob_at(ctx, pos, ctx["graph"]), image_rank(ctx, pos)
    res = {}
    for tag, sv in zip(tags[1:], s2[1:]):
        d = float(s2[0]) - float(sv); g = graph_with(ctx, D, sc, pos, e, s_orig - d)
        res[tag] = dict(d=d, rank=image_rank(ctx, pos, g), p=prob_at(ctx, pos, g))
    meta.update(s_orig=s_orig, s_mimic_full=float(s1[0]), p0=p0, rank0=r0)
    rows = []
    NA = dict(d=np.nan, rank=np.nan, p=np.nan)
    for K, S in S_of.items():
        g = lambda kind: res.get((kind, K, 0), NA)
        mr = [res[t] for t in res if t[0] == "minus_rand" and t[1] == K]; orr = [res[t] for t in res if t[0] == "only_rand" and t[1] == K]
        fp, fm = nrm(g("minus_top")["p"], p0), nrm(g("only_top")["p"], p0)
        a, b = min(max(fp, 0.), 1.), min(max(fm, 0.), 1.)
        paths = [path_of(D, anchor, e, f) for f in S]
        rows.append(dict(
            K=K, size_S=len(S), sparsity_F=1 - len(S) / max(1, meta["n_facts"]), sparsity_cand=1 - len(S) / len(cand),
            p0=p0, rank0=r0,
            fid_plus=fp, fid_minus=fm, characterization=(2 * a * (1 - b) / (a + 1 - b)) if (a + 1 - b) > 0 else np.nan,
            fid_plus_raw=p0 - g("minus_top")["p"], fid_minus_raw=p0 - g("only_top")["p"],
            dscore_minus_top=g("minus_top")["d"], drank_minus_top=g("minus_top")["rank"] - r0,
            PN=float(g("minus_top")["rank"] > 1), PS=float(g("only_top")["rank"] == 1),
            fid_perturb=nrm(g("perturb_top")["p"], p0), drank_perturb=g("perturb_top")["rank"] - r0,
            PN_perturb=float(g("perturb_top")["rank"] > 1) if K == k_full else np.nan,
            fid_plus_deg=nrm(g("minus_deg")["p"], p0), drank_deg=g("minus_deg")["rank"] - r0,
            fid_plus_rand=float(np.mean([nrm(x["p"], p0) for x in mr])), drank_rand=float(np.mean([x["rank"] - r0 for x in mr])),
            PN_rand=float(np.mean([x["rank"] > 1 for x in mr])),
            fid_minus_rand=float(np.mean([nrm(x["p"], p0) for x in orr])) if orr else np.nan,
            PS_rand=float(np.mean([x["rank"] == 1 for x in orr])) if orr else np.nan,
            on_path=float(np.mean([pp[0][0] == anchor for pp in paths])),
            heldout_overlap=int(sum(tuple(int(v) for v in D.train[f]) in D.heldout for f in S)),
            in_train=int(all(tuple(int(v) for v in D.train[f]) in D.train_set for f in S)),
            S_text=" | ".join(f"{D.ent_label[h]} –[{D.rel_label[r]}]→ {D.ent_label[t]}" for h, t, r in (D.train[f] for f in S))))
    return meta, rows
