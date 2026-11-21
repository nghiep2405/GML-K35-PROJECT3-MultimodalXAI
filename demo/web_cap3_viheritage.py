# Web demo cấp 3 (ViHeritage-MMKG, câu tiếng Việt). Nhúng nguyên văn vào demo_cap3_viheritage.ipynb, chạy sau ô X0 (mã dùng chung
# của 10b, 10c, 10d: dữ liệu 08b, checkpoint 09, hai bộ giải thích) và demo_web.py.
# Giải thích ảnh nào thì đo đúng ảnh đó: S = tối đa K* bộ ba train có δ dương lớn nhất (AdaMF: học lại nhúng kiểu Kelpie;
# GAT: che cạnh, chính xác), rồi bỏ S, làm nhiễu S, bỏ cùng số bộ ba ngẫu nhiên (10 lần) và xếp hạng lại như 10b.
import os
from types import SimpleNamespace

ARCHS = [a for a in ["gat", "adamf"] if f"{a}_full_s{MAIN_SEED}" in RUNS]
SCORER = {a: RunScorer(f"{a}_full_s{MAIN_SEED}") for a in ARCHS}
_info = XOUT / "10b_run_info.json"
K_OF = json.loads(_info.read_text(encoding="utf-8")).get("K_star", {}) if _info.exists() else {}
K_OF = {a: int(K_OF.get(a, {"adamf": 1, "gat": 8}[a])) for a in ARCHS}
MODEL_LABEL = {a: f"CLIP + {ARCH_NAME[a]}" for a in ARCHS}
VARIANTS = {"Câu hỏi": "text_vi_1", "Câu mô tả": "text_vi_2", "Tiếng Anh": "text_en"}
QT, POS, DROP = split_queries("test")
QROW = QT.set_index("query_id")
print("Mô hình:", {a: f"α_vi = {ALPHA[(SCORER[a].name, 'vi')]}, K* = {K_OF[a]}" for a in ARCHS})

def alpha_of(arch, variant):
    rs = SCORER[arch]; lang = "en" if variant == "text_en" else "vi"
    return float(ALPHA.get((rs.name, lang), rs.alpha))

def test_query(qid):
    return SimpleNamespace(query_id=qid, **QROW.loc[qid].to_dict())

# ---------- giải thích ảnh ở vị trí i (cùng thứ tự gọi bộ sinh ngẫu nhiên với explain_query của 10b) ----------
def explain_at(rs, ctx, i, K, alpha):
    g0 = rs.graph(ctx); e, a = int(ctx.ents[i]), ctx.anchor
    seed = stable_seed(ctx.qid, e, rs.name, rs.epoch); rng = np.random.default_rng(seed)
    fids, cand, n_all = universe(rs, ctx, e, rng)
    if not len(cand):
        return dict(e=e, S=[])
    delta, d_all = importance(rs, ctx, e, fids, cand, seed); order = np.argsort(-delta, kind="stable")
    S = [int(cand[j]) for j in order[:K] if delta[j] > 0]
    dS = {int(cand[j]): float(delta[j]) for j in order[:K]}
    if not S:
        return dict(e=e, S=[])
    Fset, k = [int(f) for f in fids], len(S)
    specs = [(set(S), None), (set(), {f: perturb_triple(f, keep_node(f, e, a), rng) for f in S})]
    specs += [(set(int(x) for x in rng.choice(Fset, k, replace=False)), None) for _ in range(N_RANDOM)]
    gs = variant_graphs(rs, ctx, e, fids, specs, seed, g0)
    rank = lambda g: rank_of_row(ctx.fused(g, alpha), i)
    return dict(e=e, S=S, delta=dS, r0=rank(g0), r_rm=rank(gs[0]), r_pert=rank(gs[1]), r_rand=float(np.mean([rank(g) for g in gs[2:]])),
                dscore=float(g0[i] - gs[0][i]), n_all=n_all)

def _sent(f):
    h, t, r = (int(x) for x in TRAIN[f]); return f"{LAB[h]} {RLAB[r]} {LAB[t]}"

def _arrow(f):
    h, t, r = (int(x) for x in TRAIN[f]); return f"{LAB[h]} -[{RLAB[r]}]→ {LAB[t]}"

def vh_graph(ctx, ex):
    e, a, S = ex["e"], ctx.anchor, ex["S"]
    nodes = {e: W.node(e, LAB[e], "cand", 7, f"{LAB[e]} (thực thể của ảnh)")}
    nodes[a] = W.node(a, LAB[a], "anchor", 9, f"{LAB[a]} (neo của truy vấn)", result=(a == e))
    src, dst = (a, e) if ctx.dir == "tail" else (e, a); rl = RLAB[ctx.rel]
    links = [W.link(src, dst, rl, "query", 0, f"{rl} ?", f"Quan hệ được hỏi: {LAB[src]} -[{rl}]→ {LAB[dst]}")]
    vmax = max(ex["delta"][f] for f in S); size, seen = {}, set(S)
    for f in S:
        h, t, r = (int(x) for x in TRAIN[f]); d = ex["delta"][f]
        for n in (h, t):
            nodes.setdefault(n, W.node(n, LAB[n], "other", 1.5)); size[n] = size.get(n, 0) + d
        links.append(W.link(h, t, RLAB[r], "fact", d, f"{RLAB[r]} ({d:.3g})", f"{_arrow(f)} · Δ = {d:+.4f}"))
    for f in S:
        for f2 in (path_of(f, e, a) or []):
            if f2 in seen:
                continue
            seen.add(f2); h, t, r = (int(x) for x in TRAIN[f2])
            for n in (h, t):
                nodes.setdefault(n, W.node(n, LAB[n], "other", 1.5))
            links.append(W.link(h, t, RLAB[r], "path", 0, RLAB[r], f"{_arrow(f2)} · nối về neo, không đo"))
    for n, v in size.items():
        if nodes[n]["kind"] == "other":
            nodes[n]["val"] = 1.5 + 6 * v / vmax
    f = S[0]; p = path_of(f, e, a)
    if a == e:
        story = f"Ảnh trả về chính là thực thể neo. Bằng chứng mạnh nhất: {_sent(f)}."
    elif p and len(p) > 1:
        story = f"Bằng chứng mạnh nhất: {_sent(f)}, và {_sent([x for x in p if x != f][0])} (nối về neo)."
    else:
        story = f"Bằng chứng mạnh nhất: {_sent(f)}" + (" (nối thẳng với neo)." if p else " (không nối về neo).")
    return dict(nodes=list(nodes.values()), links=links), story

# ---------- xếp hạng một truy vấn ----------
def make_ctx(q, z, gt_qid=None):
    pos = {q.query_id: POS.get(gt_qid, [])} if gt_qid else None
    drop = {q.query_id: DROP.get(gt_qid, set())} if gt_qid else None
    return QCtx(q, "test", pos, drop, z)

def rank_query(arch, q, text, gt_qid=None, k=5, variant="text_vi_1"):
    rs, K, alpha = SCORER[arch], K_OF[arch], alpha_of(arch, variant)
    ctx = make_ctx(q, encode([text])[0], gt_qid); g0 = rs.graph(ctx); f0 = ctx.fused(g0, alpha)
    order = np.argsort(-f0, kind="stable")[:int(k)]; has_gt = bool(ctx.pos.any())
    items = []
    for rank, i in enumerate(order.tolist(), 1):
        e = int(ctx.ents[i]); row = int(ctx.rows[i])
        it = dict(rank=rank, title=LAB[e], correct=bool(ctx.pos[i]) if has_gt else None, img=W.image_data_url(image_path(row)),
                  row=dict(clip_rank=rank_of_row(ctx.clip, i), removed=""),
                  more=[("Hạng nếu chỉ dùng CLIP", str(rank_of_row(ctx.clip, i))), ("Điểm đồ thị", f"{float(g0[i]):+.3f}"),
                        ("Điểm cuối", f"{float(f0[i]):+.3f}"),
                        ("Cách tính", f"{alpha:.1f}·z(CLIP) + {1 - alpha:.1f}·z(đồ thị), K* = {K}"),
                        ("Bộ giải thích", "học lại nhúng của thực thể (AdaMF)" if arch == "adamf" else "che cạnh trên đồ thị truyền tin (GAT)")])
        ex = explain_at(rs, ctx, i, K, alpha)
        if not ex["S"]:
            it["graph"] = dict(nodes=[], links=[])
            it["empty"] = "Không bộ ba train nào làm tăng điểm của thực thể này, nên không có giải thích dạng bộ ba."
        else:
            n = len(ex["S"]); r0 = ex["r0"]
            it["graph"], it["story"] = vh_graph(ctx, ex)
            it["why"] = [(f"Bỏ {n} bộ ba giải thích S", f"hạng {r0} → {W.rank_text(ex['r_rm'])}"),
                         (f"Bỏ {n} bộ ba ngẫu nhiên", f"hạng {r0} → {W.rank_text(ex['r_rand'])}")]
            it["removal"] = W.removal_sentence(n, "bộ ba", r0, ex["r_rm"], ex["r_rand"])
            it["row"]["removed"] = W.rank_text(ex["r_rm"])
            it["more"] += [("Làm nhiễu S (thay đầu kia)", f"hạng {r0} → {ex['r_pert']}"), ("Bỏ S: điểm đồ thị giảm", f"{ex['dscore']:.3f}")]
        items.append(it)
    chips = [("Neo", LAB[ctx.anchor]), ("Quan hệ", RLAB[ctx.rel] + (" (tìm đầu)" if ctx.dir == "head" else " (tìm đuôi)"))]
    target = None
    if has_gt:
        c = 1 + int((ctx.clip > ctx.clip[ctx.pos].max()).sum()); m = 1 + int((f0 > f0[ctx.pos].max()).sum())
        target = f"Ảnh đúng: CLIP hạng {c} → {MODEL_LABEL[arch]} hạng {m}"
    return dict(query=text, chips=chips, target=target), items

def rank_direct(q, text, k=5):
    """Truy vấn trực tiếp: theo thiết kế chỉ xếp hạng bằng CLIP, không có giải thích đồ thị."""
    rows = POOL["test"]; d = DROP.get(q.query_id, set())
    rows = rows[~np.isin(rows, list(d))] if d else rows
    clip = IMG_N[rows] @ encode([text])[0]; order = np.argsort(-clip, kind="stable")[:int(k)]; P = np.isin(rows, POS.get(q.query_id, []))
    items = [dict(rank=r, title=LAB[int(IMG_ENT[rows[i]])], correct=bool(P[i]) if P.any() else None, img=W.image_data_url(image_path(int(rows[i]))),
                  graph=dict(nodes=[], links=[]), row=dict(clip_rank=r, removed=""),
                  empty="Truy vấn trực tiếp: theo thiết kế chỉ xếp hạng bằng CLIP, không bộ ba nào tham gia, nên không có giải thích đồ thị.")
             for r, i in enumerate(order.tolist(), 1)]
    target = f"Ảnh đúng: CLIP hạng {1 + int((clip > clip[P].max()).sum())}" if P.any() else None
    return dict(query=text, chips=[("Loại", "trực tiếp")], target=target), items

def page(header, items):
    return W.results_iframe(header, items)

# ---------- 5 + 5 ca của báo cáo (D4) ----------
def _case_label(arch, qid):
    q = test_query(qid); text = qtext(q)
    if q.query_type == "direct":
        return f"{text}"
    rs, alpha = SCORER[arch], alpha_of(arch, "text_vi_1")
    ctx = make_ctx(q, encode([text])[0], qid); f0 = ctx.fused(rs.graph(ctx), alpha)
    if not ctx.pos.any():
        return text
    c = 1 + int((ctx.clip > ctx.clip[ctx.pos].max()).sum()); m = 1 + int((f0 > f0[ctx.pos].max()).sum())
    return f"{text}  (CLIP {c} → {m})"

DEMO_CASES = {a: {} for a in ARCHS}
_cases = XOUT / "D4_cases.csv"
if _cases.exists():
    _df = pd.read_csv(_cases)
    for a in ARCHS:
        sub = _df[(_df.arch == a) & _df.query_id.isin(QROW.index)]
        sub = pd.concat([sub[sub.loai == "thành công"], sub[sub.loai != "thành công"]])
        for r in sub.drop_duplicates("query_id").itertuples():
            DEMO_CASES[a][f"[{r.loai}] {_case_label(a, r.query_id)}"] = r.query_id
for a in ARCHS:                                            # ca Huế của hình trong báo cáo lên đầu
    hue = [c for c, q in DEMO_CASES[a].items() if "Huế có những di tích" in c]
    if hue:
        DEMO_CASES[a] = {**{hue[0]: DEMO_CASES[a][hue[0]]}, **DEMO_CASES[a]}
print({a: len(v) for a, v in DEMO_CASES.items()}, "ca trong báo cáo")

_RESULT = {}
def run_case(model, variant_name, choice, k=5):
    arch = next(a for a in ARCHS if MODEL_LABEL[a] == model); variant = VARIANTS[variant_name]
    qid = DEMO_CASES[arch].get(choice)
    if qid is None:
        return W.message_html("Chọn một ca trong danh sách.")
    key = ("case", arch, variant, qid, int(k))
    if key not in _RESULT:
        q = test_query(qid); text = qtext(q, variant)
        _RESULT[key] = page(*(rank_direct(q, text, k) if q.query_type == "direct" else rank_query(arch, q, text, qid, k, variant)))
    return _RESULT[key]

_n = lambda x: unicodedata.normalize("NFC", str(x)).lower().strip()
ENT_CHOICES = sorted({LAB[i] for i in range(N)}); REL_CHOICES = [RLAB[r] for r in sorted(RLAB)]
NAME2E = {_n(LAB[i]): i for i in range(N)}; NAME2R = {_n(RLAB[r]): r for r in RLAB}
def run_free(model, text, anchor, relation, direction, k=5):
    arch = next(a for a in ARCHS if MODEL_LABEL[a] == model)
    text = (text or "").strip(); a = NAME2E.get(_n(anchor)); r = NAME2R.get(_n(relation))
    if not text or a is None or r is None:
        return W.message_html("Nhập câu truy vấn, chọn thực thể neo và quan hệ.")
    d = "head" if direction.startswith("head") else "tail"
    hit = QT[(QT.query_type == "indirect") & (QT.anchor_entity_index == a) & (QT.relation_index == r) & (QT.prediction_direction == d)]
    gt = hit.query_id.iloc[0] if len(hit) else None              # trùng một truy vấn test: biết ảnh đúng
    key = ("free", arch, text, a, r, d, int(k))
    if key not in _RESULT:
        q = SimpleNamespace(query_id=f"demo|{text}|{a}|{r}|{d}", anchor_entity_index=a, relation_index=r, prediction_direction=d)
        variant = "text_en" if text.isascii() else "text_vi_1"
        _RESULT[key] = page(*rank_query(arch, q, text, gt, k, variant))
    return _RESULT[key]

# ---------- giao diện ----------
try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True); import gradio as gr
_m0 = MODEL_LABEL[ARCHS[0]]; _cs0 = list(DEMO_CASES[ARCHS[0]])
with gr.Blocks(title="Truy vấn đa phương thức có giải thích bằng đồ thị") as demo:
    gr.Markdown("### Cấp độ 3 · ViHeritage-MMKG (tiếng Việt, nhóm tự xây)")
    with gr.Row(equal_height=True):
        model_in = gr.Radio([MODEL_LABEL[a] for a in ARCHS], value=_m0, label="Mô hình", scale=1)
        var_in = gr.Radio(list(VARIANTS), value="Câu hỏi", label="Cách diễn đạt câu", scale=2)
        k_in = gr.Slider(3, 10, value=5, step=1, label="Top-k", scale=1)
    case_in = gr.Dropdown(_cs0, value=_cs0[0] if _cs0 else None, label="Ca trong báo cáo (5 thành công, 5 thất bại)")
    with gr.Accordion("Tự nhập truy vấn", open=False):
        with gr.Row(equal_height=True):
            text_in = gr.Textbox(label="Câu truy vấn", placeholder="Ảnh một ngôi chùa cổ ở Hà Nội", scale=4)
            anchor_in = gr.Dropdown(ENT_CHOICES, value="Huế" if "Huế" in ENT_CHOICES else ENT_CHOICES[0], label="Thực thể neo", filterable=True, scale=2)
            rel_in = gr.Dropdown(REL_CHOICES, value=REL_CHOICES[0], label="Quan hệ", filterable=True, scale=2)
            dir_in = gr.Radio(["head: (?, quan hệ, neo)", "tail: (neo, quan hệ, ?)"], value="head: (?, quan hệ, neo)", label="Chiều", scale=2)
            free_btn = gr.Button("Tìm", variant="primary", scale=1)
    out = gr.HTML()
    def _on_model(model):
        arch = next(a for a in ARCHS if MODEL_LABEL[a] == model); cs = list(DEMO_CASES[arch])
        return gr.update(choices=cs, value=cs[0] if cs else None)
    model_in.change(_on_model, model_in, case_in)
    case_args = [model_in, var_in, case_in, k_in]
    case_in.change(run_case, case_args, out); var_in.change(run_case, case_args, out); k_in.release(run_case, case_args, out)
    free_btn.click(run_free, [model_in, text_in, anchor_in, rel_in, dir_in, k_in], out)
    demo.load(run_case, case_args, out)

if os.environ.get("XAI_DEMO_LAUNCH", "1") == "1":
    W.launch(demo, IN_COLAB)
