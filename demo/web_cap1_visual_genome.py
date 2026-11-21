# Web demo cấp 1 (Visual Genome, CLIP + EdgeAwareGAT). Nhúng nguyên văn vào demo_cap1_visual_genome.ipynb, chạy sau các ô
# nạp dữ liệu, mô hình, lõi giải thích (vg_xai_core.py), chấm điểm, vg_xai_plot.py và demo_web.py.
# Giải thích = tập S gồm tối đa K* quan hệ của scene graph có Δ dương lớn nhất (Δ = điểm đồ thị giảm khi bỏ quan hệ đó),
# kiểm chứng bằng chính explain_and_measure của notebook 11: bỏ S, bỏ cùng số quan hệ ngẫu nhiên (10 lần), rồi xếp hạng lại.
import os
from collections import Counter

MAIN = f'seed_{MAIN_SEED}'
SPEC = MODELS[MAIN]
K_STAR = 12
if (OUT / '11_run_info.json').exists():
    K_STAR = int(json.loads((OUT / '11_run_info.json').read_text())['K_star'])
elif (OUT / 'V2_Kstar_val.csv').exists():
    _kd = pd.read_csv(OUT / 'V2_Kstar_val.csv'); K_STAR = int(_kd.loc[_kd.is_K_star, 'K'].iloc[0])
test_res = SPLIT['test']['res']
ROW_OF = {str(t): i for i, t in enumerate(test_res.text_id.astype(str))}

# ---------- 10 ca của báo cáo: chọn lại đúng luật V6 từ bảng hạng đã lưu (không chấm lại 20 000 truy vấn) ----------
DEMO_CASES = {}
_p = OUT / 'V6_case_pool_ranks.csv'
if _p.exists():
    rk = pd.read_csv(_p, dtype={'text_id': str, 'positive': str, 'top': str})
    def _pick(df, n, sort=None, seed=SELECTION_SEED):
        df = df.drop_duplicates('positive')
        df = df.sort_values(sort, ascending=False) if sort else df.sample(frac=1.0, random_state=seed)
        return df.head(n)
    succ = _pick(rk[(rk.r_clip > 1) & (rk.r_gat == 1) & (rk.r_no_edge > 1) & (rk.r_random > 1)].assign(gain=lambda d: d.r_clip - d.r_gat), 5, 'gain')
    harm = _pick(rk[(rk.r_clip == 1) & (rk.r_gat > 1) & ~rk.positive.isin(succ.positive)], 3)
    miss = _pick(rk[(rk.r_clip > 1) & (rk.r_gat.between(2, 5)) & ~rk.positive.isin(pd.concat([succ, harm]).positive)], 2)
    for kind, df in [('thành công', succ), ('thất bại, đồ thị làm hại', harm), ('thất bại, không cứu được', miss)]:
        for r in df.itertuples():
            text = SPLIT['test']['text'].get(r.text_id, r.text_id)
            DEMO_CASES[f'[{kind}] {text}  (CLIP {int(r.r_clip)} → {int(r.r_gat)})'] = r.text_id
print(len(DEMO_CASES), 'ca trong báo cáo | K* =', K_STAR)

# ---------- bộ mã hóa câu tự do: đúng CLIP ViT-B/32 của nhóm (kiểm tra khớp embedding đã trích) ----------
FREE_OK, FREE_MSG = False, ''
try:
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=DEVICE)
    tokenizer = open_clip.get_tokenizer('ViT-B-32'); clip_model.eval()
    @torch.no_grad()
    def encode_text(texts):
        return F.normalize(clip_model.encode_text(tokenizer(texts).to(DEVICE)).float(), dim=1).cpu()
    _chk = test_res.head(20)
    _mine = encode_text([SPLIT['test']['text'][str(t)] for t in _chk.text_id])
    _theirs = F.normalize(torch.stack([SPLIT['test']['tfeat'][SPLIT['test']['trow'][str(t)]].float() for t in _chk.text_id]), dim=1)
    FREE_OK = float((_mine * _theirs).sum(1).min()) >= 0.99
    FREE_MSG = '' if FREE_OK else 'Bộ mã hóa câu không khớp embedding của nhóm, nên tắt phần tự nhập.'
    IMG_EMB = F.normalize(torch.load(CLIP_ROOT / 'test/image_embeddings.pt', map_location='cpu', weights_only=True).float(), dim=1)
    _ii = pd.read_parquet(CLIP_ROOT / 'test/image_embedding_index.parquet')
    IMG_IDS = (_ii.sort_values('embedding_row') if 'embedding_row' in _ii else _ii).image_id.astype(str).tolist()
except Exception as exc:
    FREE_MSG = f'Không dùng được phần tự nhập: {exc}'
print('Tự nhập câu:', 'có' if FREE_OK else FREE_MSG)

# ---------- giải thích một ảnh trong top-50 ----------
def _rel_vi(r):
    return 'chứa' if r == 'contains' else r

def vg_graph(g, S, deltas):
    """Đồ thị S: nút ảnh (cam, vẽ bằng chính bức ảnh) và các vật; vật cùng tên gộp thành một nút (×n), quan hệ trùng cộng Δ."""
    lab = lambda v: 'ảnh' if v == g.root else node_label(g, v)
    seen_nodes = {}
    for u in S:
        for v in _endpoints(g, u):
            if v != g.root:
                seen_nodes.setdefault(lab(v), set()).add(v)
    items = merged_explanation(g, S, deltas, node_label, REL_NAME)
    vmax = max([t for _, _, t in items] + [1e-12]); size = Counter()
    for (s, r, d), n, t in items:
        for x in (s, d):
            if x != ROOT:
                size[x] += t
    nodes = {ROOT: W.node(ROOT, '', 'cand', 7, 'ảnh trả về')}
    for name, vs in seen_nodes.items():
        nm = name + (f' ×{len(vs)}' if len(vs) > 1 else '')
        nodes[name] = W.node(name, nm, 'other', 1.5 + 6 * size[name] / vmax, f'{nm} · tổng Δ = {size[name] * 1e3:.3g} × 10⁻³')
    links = []
    for (s, r, d), n, t in items:
        rv = _rel_vi(r) + (f' ×{n}' if n > 1 else '')
        sent = f'ảnh chứa {d}' if s == ROOT and r == 'contains' else f'{s} {r} {d}'
        links.append(W.link(s, d, rv, 'fact', t, f'{rv} ({t * 1e3:.3g})', f'{sent} · Δ = {t * 1e3:.3g} × 10⁻³'))
    story = None
    if items:
        (s, r, d), n, t = items[0]
        story = 'Bằng chứng mạnh nhất: ' + (f'ảnh chứa {d}' if s == ROOT and r == 'contains' else f'{s} {r} {d}') + (f' (×{n})' if n > 1 else '') + '.'
    return dict(nodes=list(nodes.values()), links=links), story

def vg_item(rank, j, ids, clip, gv, fused, qz, positive):
    img_id = ids[j]
    g = ComponentGraph(compact, image_to_component[img_id], image_features, entity_features, relation_type_features, DEVICE)
    m = explain_and_measure(SPEC['model'], g, qz, clip, gv, SPEC['alpha'], j, K_STAR, n_random=N_RANDOM, seed=SELECTION_SEED)
    it = dict(rank=rank, title=f'Ảnh {img_id}', correct=(img_id == positive) if positive else None,
              img=W.image_data_url(fetch_image(img_id)), row=dict(clip_rank=j + 1, removed=''),
              more=[('Hạng nếu chỉ dùng CLIP', str(j + 1)), ('Cosine CLIP', f'{float(clip[j]):.3f}'),
                    ('Điểm đồ thị', f'{float(gv[j]):+.3f}'), ('Điểm cuối', f'{float(fused[j]):+.3f}'),
                    ('Cách tính', f"{SPEC['alpha']:.1f}·z(CLIP) + {1 - SPEC['alpha']:.1f}·z(đồ thị) trên 50 ảnh CLIP, K* = {K_STAR}")])
    if not m.get('explained'):
        it['graph'] = dict(nodes=[], links=[])
        it['empty'] = 'Không quan hệ nào của ảnh làm tăng điểm đồ thị, nên không có giải thích S.'
        return it
    S = m['S']; deltas = dict(zip(S, m['deltas_S'])); r0 = m['rank0']; k = len(S)
    it['graph'], it['story'] = vg_graph(g, S, deltas)
    r_rm, r_rand = r0 + m['drank_rm'], r0 + m['drank_rand']
    it['why'] = [(f'Bỏ {k} quan hệ giải thích S', f'hạng {r0} → {W.rank_text(r_rm)}'),
                 (f'Bỏ {k} quan hệ ngẫu nhiên', f'hạng {r0} → {W.rank_text(r_rand)}')]
    it['removal'] = W.removal_sentence(k, 'quan hệ', r0, r_rm, r_rand)
    it['row']['removed'] = W.rank_text(r_rm)
    it['more'] += [('Fid+ của ảnh này', f"{m['fid_plus']:.3f} (bỏ ngẫu nhiên: {m['fid_plus_rand']:.3f})"),
                   ('Số quan hệ của ảnh', f"{m['n_units']} (S giữ {k}, sparsity {m['sparsity']:.2f})")]
    return it

LEGEND = {'cand': 'ảnh trả về', 'other': 'vật trong ảnh (to theo Δ)', 'fact': 'quan hệ trong giải thích S (dày theo Δ, số = Δ × 10⁻³)'}
ROLES = {'cand': 'ảnh hạng'}

def _page(text, ids, clip, gv, fused, qz, positive, target, k):
    order = torch.argsort(fused, descending=True).tolist()[:int(k)]
    items = [vg_item(r, j, ids, clip, gv, fused, qz, positive) for r, j in enumerate(order, 1)]
    header = dict(query=text, chips=[], target=target)
    return W.results_iframe(header, items, legend=LEGEND, roles=ROLES, removed_col='Bỏ S → hạng')

_RESULT = {}
@torch.no_grad()
def run_case_tid(tid, k=5):
    key = ('case', tid, int(k))
    if key not in _RESULT:
        row = test_res.iloc[ROW_OF[tid]]
        ids, clip, gv, qz, fused, rank, crank, top = score_row('test', MAIN, row)
        pos = str(row.positive_image_id)
        target = (f'Ảnh đúng: CLIP hạng {crank} → CLIP + GAT hạng {rank}' if pos in ids
                  else f'Ảnh đúng: CLIP hạng {crank}, ngoài top-50 nên không được xếp hạng lại')
        _RESULT[key] = _page(SPLIT['test']['text'].get(tid, tid), ids, clip, gv, fused, qz, pos, target, k)
    return _RESULT[key]

def run_case(choice, k=5):
    tid = DEMO_CASES.get(choice)
    return run_case_tid(tid, k) if tid else W.message_html('Chọn một ca trong danh sách.')

_TEXT2ID = {}
@torch.no_grad()
def run_free(text, k=5):
    text = (text or '').strip()
    if not text:
        return W.message_html('Nhập câu truy vấn tiếng Anh.')
    if not _TEXT2ID:
        _TEXT2ID.update({v.strip(): t for t, v in SPLIT['test']['text'].items()})
    if text in _TEXT2ID and _TEXT2ID[text] in ROW_OF:           # câu trùng một truy vấn test: dùng đúng truy vấn đó để biết ảnh đúng
        return run_case_tid(_TEXT2ID[text], k)
    if not FREE_OK:
        return W.message_html(FREE_MSG)
    key = ('free', text, int(k))
    if key not in _RESULT:
        q = encode_text([text])[0]; clip_all = IMG_EMB @ q
        top50 = torch.topk(clip_all, 50).indices.tolist(); ids = [IMG_IDS[j] for j in top50]
        qz = SPEC['model'].encode_query(q.unsqueeze(0).to(DEVICE))[0]
        gv = (split_emb('test', MAIN)[[SPLIT['test']['row'][x] for x in ids]] @ qz.cpu()).float()
        clip = clip_all[top50]; fused = fuse(clip, gv, SPEC['alpha'])
        _RESULT[key] = _page(text, ids, clip, gv, fused, qz, None, None, k)
    return _RESULT[key]

# ---------- giao diện ----------
try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'], check=True); import gradio as gr
_cs = list(DEMO_CASES)
with gr.Blocks(title='Truy vấn đa phương thức có giải thích bằng đồ thị') as demo:
    gr.Markdown('### Cấp độ 1 · CLIP + GAT trên scene graph Visual Genome')
    with gr.Row(equal_height=True):
        case_in = gr.Dropdown(_cs, value=_cs[0] if _cs else None, label='Ca trong báo cáo (5 thành công, 5 thất bại)', scale=6)
        k_in = gr.Slider(3, 10, value=5, step=1, label='Top-k', scale=1)
    with gr.Accordion('Tự nhập truy vấn', open=False):
        with gr.Row(equal_height=True):
            text_in = gr.Textbox(label='Câu truy vấn (tiếng Anh)', placeholder='a man riding a horse on the beach', scale=6)
            free_btn = gr.Button('Tìm', variant='primary', scale=1)
    out = gr.HTML()
    case_in.change(run_case, [case_in, k_in], out); k_in.release(run_case, [case_in, k_in], out)
    free_btn.click(run_free, [text_in, k_in], out); text_in.submit(run_free, [text_in, k_in], out)
    demo.load(run_case, [case_in, k_in], out)

if os.environ.get('XAI_DEMO_LAUNCH', '1') == '1':
    W.launch(demo, 'google.colab' in sys.modules)
