# C10. Demo web cấp 2 (DB15K, FB15K-237-IMG): giao diện dùng chung multimodal_xai.demo_web, giống demo cấp 1 và cấp 3.
# Chạy riêng demo: chạy ô C0, C0b, C0c rồi chạy thẳng hai ô C10 (không cần C1 đến C9; K* và 10 ca đọc từ kết quả đã lưu).
import html as _html
from types import SimpleNamespace
from multimodal_xai import demo_web as W
from multimodal_xai.workflows.runner import run_shared

if "Split" not in globals():                    # chưa chạy C1: nạp phần xếp hạng dùng chung
    _TEXT_CACHE = {}
    run_shared("ranking", globals())
DATA_BY_KIND = {DATA[n].kind: DATA[n] for n in DATA}
if "explain_facts" not in globals():            # chưa chạy C4
    run_shared("explain", globals())
MOD = ["cấu trúc (KG)", "ảnh", "văn bản"]
MODEL_NAME = "CLIP + AdaMF"

def _result_file(name, fname):
    for d in [DATA[name].out, FINAL / "results" / name / "XAI"]:
        p = Path(d) / fname
        if p.exists():
            return p
    return None

if "K_STAR" not in globals():
    K_STAR = {}
for _n in DATASET_NAMES:
    if _n not in K_STAR:
        _p = _result_file(_n, "C6_K_star.json")
        K_STAR[_n] = int(json.loads(_p.read_text())["K_star"]) if _p else {"DB15K": 2, "FB15K": 8}.get(_n, 2)

if "explain_image" not in globals():            # giống C8 (18.py)
    def explain_image(ctx, D, sc, pos):
        ex = explain_modality(ctx, D, sc, pos)
        if ctx["q"].prediction_direction != "direct":
            ex["E3"] = explain_facts(ctx, D, sc, pos, [K_STAR[D.name]])
        return ex

# ---------- 10 ca của báo cáo (C7) ----------
CASE_KIND = {"thanh_cong": "thành công", "that_bai_bi_lam_hai": "thất bại, đồ thị làm hại",
             "that_bai_khong_cuu_duoc": "thất bại, không cứu được"}
REPORT_FIGURE_CASE = {"DB15K": "q_ffbbb7ed71136914b34d1caf"}      # ca Hitchens / Richard Dawkins của hình trong báo cáo
DEMO_CASES = {}
for _n in DATASET_NAMES:
    _df = CASES[_n] if "CASES" in globals() and _n in CASES else None
    if _df is None:
        _p = _result_file(_n, "C7_cases.csv")
        _df = pd.read_csv(_p) if _p else pd.DataFrame(columns=["case", "query_id", "query_text", "clip_rank", "hybrid_rank"])
    _df = _df[_df.query_id.isin(set(DATA[_n].test_q.query_id))]
    first = REPORT_FIGURE_CASE.get(_n)
    _df = pd.concat([_df[_df.query_id == first], _df[_df.query_id != first]]) if first in set(_df.query_id) else _df
    DEMO_CASES[_n] = {f"[{CASE_KIND.get(r.case, r.case)}] {r.query_text}  (CLIP {int(r.clip_rank)} → {int(r.hybrid_rank)})": r.query_id
                      for r in _df.itertuples()}

for D in DATA.values():
    lab = pd.Series(D.ent_label); dup = lab.duplicated(keep=False)
    D.ent_choice = [f"{l} [{i}]" if dup[i] else l for i, l in enumerate(D.ent_label)]
    D.choice_to_ent = {c: i for i, c in enumerate(D.ent_choice)}
    rl = pd.Series(D.rel_label); dupr = rl.duplicated(keep=False)
    D.rel_choice = [f"{l} [{i}]" if dupr[i] else l for i, l in enumerate(D.rel_label)]
    D.choice_to_rel = {c: i for i, c in enumerate(D.rel_choice)}

# ---------- ngữ cảnh xếp hạng ----------
_SPLIT_ONE = {}
def test_ctx(name, qid):
    """Truy vấn test có sẵn: xếp hạng y như lúc đánh giá (CLIP trên cả pool test, top-1000, rerank), có mặt nạ ảnh đúng."""
    D, sc = DATA[name], SCORERS[(name, MAIN_SEED)]
    if (name, qid) not in _SPLIT_ONE:
        _SPLIT_ONE[(name, qid)] = Split(name, "test", D.test_q[D.test_q.query_id == qid].reset_index(drop=True))
    return dict(_SPLIT_ONE[(name, qid)].ctx(0, sc), gt=True)

def free_ctx(name, q, text):
    """Truy vấn tự nhập không có trong test: cùng cách xếp hạng, nhưng không biết ảnh đúng."""
    D, sc = DATA[name], SCORERS[(name, MAIN_SEED)]
    tf = encode_texts(name, [text]); pool = D.pool["test"]
    feats = F.normalize(torch.tensor(np.asarray(D.img_feat[pool]), dtype=torch.float32), dim=1).to(DEVICE)
    s = (tf.to(DEVICE) @ feats.T)[0]; order = torch.argsort(s, descending=True, stable=True)[:min(TOPK, len(pool))]
    rows = torch.as_tensor(pool[order.cpu().numpy()]).long()
    qp = sc.qproj(tf)[0] if sc.kind == "fb" else None
    g, ent = sc.image_scores(q, rows, D, qproj=qp)
    return dict(q=q, i=0, rows=rows, clip=s[order], graph=g, ent=ent, qproj=qp, base=1,
                tmask=torch.zeros(len(rows), dtype=torch.bool, device=DEVICE), alpha=sc.alpha, kind=sc.kind)

def query_text_for(D, anchor, rel, direction):
    allq = pd.concat([D.test_q, D.val_q])
    hit = allq[(allq.prediction_direction == direction) & (allq.anchor_entity_index == anchor) & (allq.relation_index == rel)]
    if len(hit):
        return hit.query_text.iloc[0]
    rl, al = D.rel_label[rel], D.ent_label[anchor]
    return (f"Find an image of the entity that is related to {al} by {rl}." if direction == "tail"
            else f"Find an image of the entity whose {rl} relation points to {al}.")

def thumb(D, row, size=640):
    return W.image_data_url(open_image(D, row), size)

def _g(x):
    return f"{x:.3g}"

def kg_graph(D, q, ex, K):
    """Đồ thị giải thích: neo, thực thể của ảnh, K* bộ ba train có Δ dương (nét đậm) và bộ ba nối chúng về neo (nét mảnh)."""
    nodes, links, seen = {}, [], set()
    e = ex["entity"]; nodes[e] = W.node(e, D.ent_label[e], "cand", 7, f"{D.ent_label[e]} (thực thể của ảnh)")
    if q.prediction_direction != "direct":
        a = int(q.anchor_entity_index); rl = D.rel_label[int(q.relation_index)]
        nodes[a] = W.node(a, D.ent_label[a], "anchor", 9, f"{D.ent_label[a]} (neo của truy vấn)", result=(a == e))
        src, dst = (a, e) if q.prediction_direction == "tail" else (e, a)
        links.append(W.link(src, dst, rl, "query", 0, f"{rl} ?", f"Quan hệ được hỏi: {D.ent_label[src]} -[{rl}]→ {D.ent_label[dst]}"))
    facts = [f for f in ex.get("E3", {}).get("facts", [])[:K] if f["delta"] > 0]
    vmax = max([f["delta"] for f in facts] + [1e-12])
    for main in (True, False):                     # bộ ba giải thích trước, bộ ba nối về neo sau
        for f in facts:
            for (_, _, fid) in f["path"]:
                if (fid == f["fid"]) != main:
                    continue
                h, t, r = (int(x) for x in D.train[fid])
                if (h, t, r) in seen:
                    continue
                seen.add((h, t, r))
                for n in (h, t):
                    nodes.setdefault(n, W.node(n, D.ent_label[n], "other", 1.5))
                lab = D.rel_label[r]
                links.append(W.link(h, t, lab, "fact" if main else "path", f["delta"] if main else 0,
                                    f"{lab} ({_g(f['delta'])})" if main else lab,
                                    f"{D.ent_label[h]} -[{lab}]→ {D.ent_label[t]}" + (f" · Δ = {f['delta']:+.4f}" if main else " · nối về neo, không đo")))
            if main:
                x = f["head"] if f["tail"] == e else f["tail"]
                if nodes[x]["kind"] == "other":
                    nodes[x]["val"] = 1.5 + 6 * f["delta"] / vmax
    return dict(nodes=list(nodes.values()), links=links)

def kg_story(D, q, ex, K):
    """Một câu trả lời "Vì sao trả về X?": bộ ba có Δ lớn nhất và cách nó nối về thực thể neo."""
    facts = [f for f in ex.get("E3", {}).get("facts", [])[:K] if f["delta"] > 0]
    if not facts:
        return None
    sent = lambda fid: " ".join([D.ent_label[int(D.train[fid][0])], D.rel_label[int(D.train[fid][2])], D.ent_label[int(D.train[fid][1])]])
    f, e, a = facts[0], ex["entity"], int(q.anchor_entity_index)
    if a == e:
        return f"Ảnh trả về chính là thực thể neo. Bằng chứng mạnh nhất: {sent(f['fid'])}."
    if len(f["path"]) == 2:
        return f"Bằng chứng mạnh nhất: {sent(f['fid'])}, và {sent(f['path'][0][2])} (nối về neo)."
    if f["path"][0][0] == a:
        return f"Bằng chứng mạnh nhất: {sent(f['fid'])} (nối thẳng với neo)."
    return f"Bằng chứng mạnh nhất: {sent(f['fid'])} (không nối về neo)."

def search_and_explain(name, ctx, k=5):
    D, sc = DATA[name], SCORERS[(name, MAIN_SEED)]; q = ctx["q"]; K = K_STAR[name]
    graph_used = not (ctx["graph"] is None or (sc.kind == "db" and (q.prediction_direction == "direct" or sc.alpha == 1.0)))
    fused = sc.alpha * zscore(ctx["clip"]) + (1 - sc.alpha) * zscore(ctx["graph"]) if graph_used else ctx["clip"]
    has_gt = bool(ctx.get("gt", False))
    items = []
    for rank, p in enumerate(top_order(ctx)[:k].tolist(), 1):
        row = int(ctx["rows"][p]); e = int(ctx["ent"][p]) if ctx["ent"] is not None else int(D.img_ent[row, 0])
        it = dict(rank=rank, title=D.ent_label[e], correct=bool(ctx["tmask"][p]) if has_gt else None, img=thumb(D, row),
                  graph=dict(nodes=[], links=[]), row=dict(clip_rank=p + 1, removed=""),
                  more=[("Hạng nếu chỉ dùng CLIP", str(p + 1)), ("Điểm CLIP", f"{float(ctx['clip'][p]):.3f}")])
        if not graph_used:
            it["empty"] = (f"{name}, truy vấn trực tiếp: theo thiết kế, loại truy vấn này chỉ xếp hạng bằng CLIP, "
                           "không cạnh hay bộ ba nào tham gia, nên không có giải thích đồ thị.")
            items.append(it); continue
        ex = explain_image(ctx, D, sc, p)
        it["more"] += [("Điểm đồ thị", f"{float(ctx['graph'][p]):+.3f}"), ("Điểm cuối", f"{float(fused[p]):+.3f}"),
                       ("Cách tính", f"{sc.alpha:.1f}·z(CLIP) + {1 - sc.alpha:.1f}·z(đồ thị), K* = {K}")]
        it["bars"] = [dict(label=MOD[m], w=ex["weights"][m],
                           text=f"w {ex['weights'][m]:.2f}" + (f" · che: điểm giảm {ex['drop'][m]['delta']:.2f}, hạng → {ex['drop'][m]['rank']}"
                                                              if m in ex["drop"] else "")) for m in range(3) if ex["present"][m]]
        it["graph"] = kg_graph(D, q, ex, K)
        E3 = ex.get("E3")
        if q.prediction_direction == "direct":
            it["empty"] = ("FB15K-237-IMG, truy vấn trực tiếp: điểm đồ thị là độ khớp giữa câu và nhúng của thực thể, không đi qua "
                           "quan hệ nào, nên không vẽ bộ ba. Đóng góp theo phương thức nằm trong mục Chi tiết.")
            it["graph"] = dict(nodes=[], links=[])
        elif E3 and E3.get("ok") and f"top@{K}" in E3.get("removal", {}):
            nS = len([f for f in E3["facts"][:K] if f["delta"] > 0]); r0 = E3["rank_full"]
            top, rnd = E3["removal"][f"top@{K}"], E3["removal"].get(f"rand@{K}")
            it["why"] = [(f"Bỏ {nS} bộ ba giải thích S", f"hạng {r0} → {W.rank_text(r0 + top['drank'])}")]
            it["row"]["removed"] = W.rank_text(r0 + top["drank"])
            it["more"].append(("Bỏ S: điểm đồ thị giảm", f"{top['dscore']:.3f}"))
            if rnd:
                it["why"].append((f"Bỏ {nS} bộ ba ngẫu nhiên", f"hạng {r0} → {W.rank_text(r0 + rnd['drank'])}"))
                it["more"].append((f"Bỏ ngẫu nhiên: điểm đồ thị giảm", f"{rnd['dscore']:.3f} (trung bình {N_RANDOM} lần)"))
            it["story"] = kg_story(D, q, ex, K)
            it["removal"] = W.removal_sentence(nS, "bộ ba", r0, r0 + top["drank"], r0 + rnd["drank"] if rnd else None)
        else:
            it["graph"] = dict(nodes=[], links=[])
            it["empty"] = "Thực thể của ảnh không có bộ ba train nào làm tăng điểm, nên không có giải thích dạng bộ ba."
        items.append(it)
    return items, has_gt, graph_used

def page_header(name, ctx, has_gt):
    D = DATA[name]; q = ctx["q"]
    if q.prediction_direction == "direct":
        chips = [("Loại", "trực tiếp")]
    else:
        chips = [("Neo", D.ent_label[int(q.anchor_entity_index)]),
                 ("Quan hệ", D.rel_label[int(q.relation_index)] + (" (tìm đầu)" if q.prediction_direction == "head" else " (tìm đuôi)"))]
    target = None
    if has_gt and bool(ctx["tmask"].any()):
        target = f"Ảnh đúng: CLIP hạng {ctx['base']} → {MODEL_NAME} hạng {rank_in(ctx)}"
    elif has_gt:
        target = f"Ảnh đúng: CLIP hạng {ctx['base']}, ngoài top-{len(ctx['rows'])} nên không được xếp hạng lại"
    return dict(query=q.query_text, chips=chips, target=target)

_RESULT = {}
def render(name, ctx, k):
    items, has_gt, _ = search_and_explain(name, ctx, int(k))
    header = page_header(name, ctx, has_gt)
    with open(DATA[name].out / "demo_explanations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(dataset=name, query=header["query"], top=[dict(rank=i["rank"], entity=i["title"]) for i in items]),
                           ensure_ascii=False) + "\n")
    return header, items

def run_case(name, choice, k=5):
    qid = DEMO_CASES.get(name, {}).get(choice)
    if qid is None:
        return W.message_html("Chọn một ca trong danh sách.")
    key = (name, qid, int(k))
    if key not in _RESULT:
        _RESULT[key] = W.results_iframe(*render(name, test_ctx(name, qid), k))
    return _RESULT[key]

def run_free(name, mode, anchor_choice, rel_choice, direction_choice, free_text, k=5):
    D = DATA[name]
    if mode.startswith("Gián tiếp"):
        if anchor_choice not in D.choice_to_ent or rel_choice not in D.choice_to_rel:
            return W.message_html("Chọn thực thể neo và quan hệ.")
        a, r = D.choice_to_ent[anchor_choice], D.choice_to_rel[rel_choice]; d = "tail" if direction_choice.startswith("tail") else "head"
        tq = D.test_q[(D.test_q.prediction_direction == d) & (D.test_q.anchor_entity_index == a) & (D.test_q.relation_index == r)]
        if len(tq):                                # trùng một truy vấn test: dùng đúng truy vấn đó để biết ảnh đúng
            return run_case_qid(name, tq.query_id.iloc[0], k)
        text = query_text_for(D, a, r, d)
        q = SimpleNamespace(prediction_direction=d, anchor_entity_index=a, relation_index=r, query_text=text, query_id=f"demo|{a}|{r}|{d}")
    else:
        text = (free_text or "").strip()
        if not text:
            return W.message_html("Nhập câu truy vấn tiếng Anh.")
        tq = D.test_q[(D.test_q.prediction_direction == "direct") & (D.test_q.query_text.str.strip() == text)]
        if len(tq):
            return run_case_qid(name, tq.query_id.iloc[0], k)
        q = SimpleNamespace(prediction_direction="direct", anchor_entity_index=None, relation_index=None, query_text=text, query_id=f"demo|{text}")
    key = (name, q.query_id, int(k))
    if key not in _RESULT:
        _RESULT[key] = W.results_iframe(*render(name, free_ctx(name, q, text), k))
    return _RESULT[key]

def run_case_qid(name, qid, k=5):
    key = (name, qid, int(k))
    if key not in _RESULT:
        _RESULT[key] = W.results_iframe(*render(name, test_ctx(name, qid), k))
    return _RESULT[key]

# Xem trước trong notebook: ca đầu danh sách của tập đầu tiên (DB15K: ca Hitchens trong báo cáo)
_n0 = DATASET_NAMES[0]; _c0 = next(iter(DEMO_CASES[_n0]), None)
if _c0 is not None and os.environ.get("XAI_DEMO_PREVIEW", "1") == "1":
    _h0, _items0 = render(_n0, test_ctx(_n0, DEMO_CASES[_n0][_c0]), 5)
    (DATA[_n0].out / "figures").mkdir(parents=True, exist_ok=True)
    (DATA[_n0].out / "figures" / "demo_preview.html").write_text(W.results_page(_h0, _items0), encoding="utf-8")
    display(HTML(W.results_iframe(_h0, _items0)))
