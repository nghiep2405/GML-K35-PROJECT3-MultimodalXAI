"""Sinh notebook 11_visualgenome_gat_xai_level1.ipynb (nhúng nguyên văn vg_xai_core.py).

Chạy: python src/notebooks/build_vg_xai_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = (HERE / "vg_xai_core.py").read_text(encoding="utf-8")
PLOT = (HERE / "vg_xai_plot.py").read_text(encoding="utf-8")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# 11 · XAI cấp độ 1 trên Visual Genome: giải thích xếp hạng của CLIP + EdgeAwareGAT

Đề, mục 5 (cấp 1) và mục 7 (Nhiệm vụ 3): hiển thị quan hệ hoặc đường đi được dùng khi xếp hạng cho một số truy vấn,
phân tích 5 truy vấn thành công và 5 truy vấn thất bại; giải thích phải đến từ chính cơ chế xếp hạng (mục 2).
Notebook này còn đo fidelity, validity, sparsity theo cùng định nghĩa với notebook 10a, 10b để có dòng Visual Genome
trong bảng XAI chung.

**Mô hình được giải thích:** đúng các checkpoint đã khóa ở `results/VisualGenome/3SEED/seed_{2026,2027,2028}/best_gat.pt`,
α lấy từ checkpoint (chọn trên validation). Không huấn luyện lại, không chọn gì trên test.

**Bộ giải thích (giống notebook 06 của nhóm, áp dụng cho scene graph):** đơn vị là một quan hệ của ảnh, gồm cạnh gốc và
cạnh ngược (cạnh ảnh → thực thể `contains` cũng là đơn vị). Đóng góp Δ\_u = S\_full − S\_(full \ u), trong đó S là điểm đồ thị
q'·z của ảnh, z được tính lại bằng chính 2 tầng GAT của checkpoint sau khi bỏ u. Giải thích S = tối đa K\* đơn vị có Δ dương
lớn nhất. Attention của GAT chỉ dùng làm đường cơ sở: bộ mã hóa đồ thị không nhận truy vấn nên attention giống nhau cho mọi truy vấn.

| Bước | Nội dung | File kết quả (`0. final/results/VisualGenome/XAI/`) |
|---|---|---|
| V1 | Tái lập MRR test của seed 2026 từ checkpoint, đối chiếu `final_test_summary.json` | `V1_replay_check.json` |
| V2 | Chọn K\* trên **validation** (K nhỏ nhất có trung vị D(K)/D\_all ≥ 0,8) | `V2_Kstar_val.csv` |
| V3 | Test, 3 seed: giải thích ảnh hạng 1 trả về; bỏ / chỉ giữ / làm nhiễu S; so với K ngẫu nhiên và K theo attention | `V3_fidelity_per_query.csv`, `V3_fidelity_summary.csv`, `V3_tests.csv` |
| V4 | Đường cong theo K (seed chính) | `V4_sparsity_K_curve.csv` |
| V5 | Validity: phản thực (PN), vùng tiếp nhận, khớp nội dung truy vấn, ổn định giữa seed | `V5_validity.csv` |
| V6 | 5 thành công + 5 thất bại, có hình | `V6_cases.md`, `cases/*.png` |
| V7 | Demo: nhập câu truy vấn → top-k ảnh + đồ thị giải thích | `demo_*.png` |
| — | Dòng cho bảng XAI chung | `XAI_L3_row.csv` |

**Định nghĩa** (p = softmax của điểm cuối α·z(CLIP) + (1−α)·z(đồ thị) trên 50 ứng viên CLIP của truy vấn; ảnh được giải thích là
ảnh hạng 1 hệ thống trả về; F = mọi đơn vị trong đồ thị của ảnh):

| Chỉ số | Cách tính | Hướng tốt |
|---|---|---|
| Fid+ (cần thiết) | (p − p₋ₛ) / p khi bỏ S | ↑ |
| Fid− (đủ) | (p − pₛ) / p khi chỉ giữ S | ↓ |
| Characterization | 2·Fid+·(1 − Fid−) / (Fid+ + 1 − Fid−), sau khi kẹp Fid± về [0, 1] | ↑ |
| Fid làm nhiễu | (p − p_pert) / p khi thay đầu xa nút ảnh của mỗi cạnh trong S bằng một thực thể ngẫu nhiên khác của ảnh | ↑ |
| PN | % truy vấn mà bỏ S làm ảnh mất hạng 1 | ↑ |
| PS | % truy vấn mà chỉ giữ S ảnh vẫn hạng 1 | ↑ |
| Sparsity | 1 − \|S\| / \|F\| | ↑ (khi fidelity vẫn đạt) |
| Validity cấu trúc | (i) S chỉ gồm quan hệ của chính scene graph của ảnh; (ii) % đơn vị nằm trong vùng tiếp nhận 2 tầng | (i) = 100%, (ii) = 100% |
| Khớp nội dung | % đơn vị của S chạm thực thể có từ xuất hiện trong câu truy vấn, so với K ngẫu nhiên và tỉ lệ nền | ↑ |
| Ổn định | Jaccard của S giữa các seed khi ba seed trả về cùng ảnh | ↑ |

Fid± theo khảo sát XAI cho GNN (Amara và cs., arXiv 2207.12599). Đường cơ sở: K đơn vị ngẫu nhiên (10 lần) và K đơn vị
có attention lớn nhất.

**Chạy:** Colab GPU (T4 đủ) → Run all. Cần lối tắt `MLGs - Project` trong My Drive. V1 chạy toàn bộ 523.347 truy vấn test
(vài chục phút); đặt `RUN_FULL_REPLAY = False` để bỏ qua. Kết quả từng seed ở V3 được ghi ngay, phiên bị ngắt thì Run all lại.
""")

code(r"""
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'torch-geometric', 'pyarrow', 'pandas', 'scipy', 'networkx', 'open_clip_torch', 'requests'])
print('Đã cài thư viện')
""")

md("## V0. Tìm `0. final` trên Drive, cấu hình")
code(r"""
import os, json, glob, time, random, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('Không chạy trên Colab:', exc)

def find_final_root():
    env = os.environ.get('FINAL_ROOT')
    if env and Path(env).is_dir():
        return Path(env)
    pats = ['/content/drive/MyDrive/MLGs - Project/0. final',
            '/content/drive/MyDrive/*/MLGs - Project/0. final',
            '/content/drive/MyDrive/*/*/MLGs - Project/0. final',
            '/content/drive/Shareddrives/*/MLGs - Project/0. final']
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError('Không tìm thấy "MLGs - Project/0. final". Thêm lối tắt vào My Drive hoặc đặt FINAL_ROOT.')

FINAL = find_final_root()
DATA_ROOT = FINAL / 'data/VisualGenome/processed'
RESULTS_ROOT = FINAL / 'results/VisualGenome'
GRAPH_ROOT, CLIP_ROOT = DATA_ROOT / 'graph', DATA_ROOT / 'clip'
FEATURE_ROOT = CLIP_ROOT / 'features'
OUT = RESULTS_ROOT / 'XAI'
(OUT / 'cases').mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAIN_SEED = 2026
SEEDS = [2026, 2027, 2028]
SELECTION_SEED = 2026          # bốc truy vấn cố định, không nhìn kết quả
N_VAL_K = 300                  # số truy vấn val để chọn K*
N_TEST = 500                   # số truy vấn test đo fidelity mỗi seed
N_RANDOM = 10                  # số lần bỏ K đơn vị ngẫu nhiên
K_GRID = [1, 2, 3, 5, 8, 12, 16]
N_CASE_POOL = 20000            # số truy vấn test dùng để chọn ca 5 + 5
RUN_FULL_REPLAY = True
print('FINAL =', FINAL, '| DEVICE =', DEVICE)
""")

md("## Nạp đồ thị, đặc trưng, tên thực thể và tên quan hệ (giống `evaluate_test.ipynb`)")
code(r"""
compact = torch.load(DATA_ROOT / 'graph_cache/compact_graph_cache_v2.pt', map_location='cpu', weights_only=False)
assert compact['node_ptr'][-1].item() == 2_609_364 and compact['edge_ptr'][-1].item() == 8_005_044
image_features = torch.load(FEATURE_ROOT / 'image_features.pt', map_location='cpu', weights_only=True).float()
entity_features = torch.load(FEATURE_ROOT / 'entity_label_features.pt', map_location='cpu', weights_only=True).float()
relation_features = torch.load(FEATURE_ROOT / 'relation_text_features.pt', map_location='cpu', weights_only=True).float()
relation_mapping = pd.read_parquet(GRAPH_ROOT / 'relation_mapping.parquet')
relation_feature_index = pd.read_parquet(FEATURE_ROOT / 'relation_feature_index.parquet')
relation_join = relation_mapping.merge(relation_feature_index[['edge_family', 'relation', 'feature_row']],
                                       on=['edge_family', 'relation'], how='left', validate='one_to_one')
relation_type_features = torch.empty((int(relation_mapping.relation_index.max()) + 1, 512), dtype=torch.float32)
known = relation_join.feature_row.notna().to_numpy()
relation_type_features[torch.from_numpy(relation_join.loc[known, 'relation_index'].to_numpy(np.int64))] = \
    relation_features[torch.from_numpy(relation_join.loc[known, 'feature_row'].to_numpy(np.int64))]
unk_rows = relation_feature_index.loc[relation_feature_index.learnable_relation_index == 1, 'feature_row'].to_numpy(np.int64)
unk_feature = F.normalize(relation_features[torch.from_numpy(unk_rows)].mean(0), dim=0)
missing = relation_join.loc[~known, 'relation_index'].to_numpy(np.int64)
assert set(missing).issubset({1})
if len(missing):
    relation_type_features[torch.from_numpy(missing)] = unk_feature
REL_NAME = dict(zip(relation_mapping.relation_index, relation_mapping.relation))

image_index = pd.read_parquet(FEATURE_ROOT / 'image_feature_index.parquet', columns=['image_id', 'feature_row'])
image_to_component = dict(zip(image_index.image_id.astype(str), image_index.feature_row))

def pick_text_col(df, prefer):
    cols = [c for c in df.columns
            if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    assert cols, f'Không có cột chữ trong {list(df.columns)}'
    pref = [c for c in cols if any(k in c.lower() for k in prefer)]
    return (pref or cols)[0]

efi = pd.read_parquet(FEATURE_ROOT / 'entity_feature_index.parquet')
print('entity_feature_index:', list(efi.columns))
ENT_LABEL_COL = pick_text_col(efi, ('label', 'name', 'canonical', 'text'))
ENT_LABEL = (dict(zip(efi.feature_row, efi[ENT_LABEL_COL])) if 'feature_row' in efi.columns
             else dict(enumerate(efi[ENT_LABEL_COL])))
print('Dùng cột tên thực thể:', ENT_LABEL_COL, '| ví dụ:', list(ENT_LABEL.values())[:8])

def node_label(g, v):
    return 'ẢNH' if v == g.root else str(ENT_LABEL.get(int(g.feature_rows[v]), f'thực thể {v}'))

def load_queries(split):
    q = pd.read_parquet(CLIP_ROOT / split / 'queries.parquet')
    col = pick_text_col(q.drop(columns=[c for c in q.columns if c.endswith('_id')], errors='ignore'),
                        ('text', 'phrase', 'caption', 'query', 'description'))
    return dict(zip(q.text_id.astype(str), q[col].astype(str)))

IMG_URL = {}
try:
    imgs = pd.read_parquet(DATA_ROOT / 'splits/images.parquet')
    ucol = [c for c in imgs.columns if 'url' in c.lower()]
    if ucol:
        IMG_URL = dict(zip(imgs.image_id.astype(str), imgs[ucol[0]].astype(str)))
    print('images.parquet:', list(imgs.columns), '| có URL ảnh:', bool(ucol))
except Exception as exc:
    print('Không đọc được URL ảnh (chỉ vẽ đồ thị):', exc)
print('Đã nạp đặc trưng')
""")

md("## Mô hình và checkpoint đã khóa")
code(r"""
class EdgeAwareGAT(nn.Module):
    # Giữ nguyên kiến trúc của run_gat.py / evaluate_test.ipynb
    def __init__(self, input_dim=512, hidden_dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.edge_projection = nn.Linear(input_dim + 1, hidden_dim)
        self.type_embedding = nn.Embedding(2, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim, dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim, dropout=dropout, add_self_loops=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_graph(self, batch):
        h = self.node_projection(batch['x']) + self.type_embedding(batch['node_type'])
        edge_attr = self.edge_projection(torch.cat([batch['edge_attr'], batch['edge_reverse']], dim=1))
        update = self.gat1(h, batch['edge_index'], edge_attr=edge_attr)
        h = self.norm1(h + F.dropout(F.elu(update), self.dropout, self.training))
        update = self.gat2(h, batch['edge_index'], edge_attr=edge_attr)
        h = self.norm2(h + F.dropout(F.elu(update), self.dropout, self.training))
        return F.normalize(h[batch['root_local']], dim=1)

    def encode_query(self, q):
        return F.normalize(self.query_projection(q), dim=1)

RUNS = {
    'seed_2026': ('3SEED/seed_2026', 'real_edges'),
    'seed_2027': ('3SEED/seed_2027', 'real_edges'),
    'seed_2028': ('3SEED/seed_2028', 'real_edges'),
    'no_edge': ('ABLATION/no_edge_seed_2026', 'no_edge'),
    'random_edges': ('ABLATION/random_edges_seed_2026', 'random_edges'),
}
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

MODELS = {}
for name, (rel, mode) in RUNS.items():
    path = RESULTS_ROOT / rel / 'best_gat.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('edge_mode', 'real_edges') == mode, (name, ck.get('edge_mode'))
    m = EdgeAwareGAT().to(DEVICE)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    MODELS[name] = {'model': m, 'alpha': float(ck['alpha']), 'epoch': int(ck['epoch']), 'edge_mode': mode,
                    'sha256': sha256(path)}
    print(f"{name}: epoch {ck['epoch']}, α_CLIP = {ck['alpha']}, edge_mode = {mode}")
""")

md("## Lõi bộ giải thích (nhúng nguyên văn `vg_xai_core.py`, đã kiểm thử khớp mã gốc trên dữ liệu giả lập)")
code(CORE)

md("## Chấm điểm trên 50 ứng viên CLIP của mỗi truy vấn (giống `evaluate_frozen_test`)")
code(r"""
global_to_local = torch.full((2_609_364,), -1, dtype=torch.long)

def build_component_batch(component_rows, edge_mode):
    node_chunks, edge_chunks, rel_chunks, rev_chunks, roots = [], [], [], [], []
    for comp in component_rows:
        ns, ne = compact['node_ptr'][comp:comp + 2].tolist()
        es, ee = compact['edge_ptr'][comp:comp + 2].tolist()
        nodes = compact['node_perm'][ns:ne]
        edges = compact['edge_index'][:, es:ee]
        if edge_mode == 'random_edges':
            gen = torch.Generator().manual_seed(2026 + comp)
            perm = torch.randperm(edges.shape[1], generator=gen)
            edges = torch.stack((edges[0], edges[1, perm]))
        nt = compact['node_type'][nodes]
        roots.append(nodes[nt == 1].item())
        node_chunks.append(nodes); edge_chunks.append(edges)
        rel_chunks.append(compact['edge_relation_type'][es:ee]); rev_chunks.append(compact['edge_is_reverse'][es:ee])
    ng = torch.cat(node_chunks); eg = torch.cat(edge_chunks, 1)
    global_to_local[ng] = torch.arange(ng.numel())
    el = global_to_local[eg]; rl = global_to_local[torch.tensor(roots)]
    nt = compact['node_type'][ng].long(); fr = compact['node_feature_row'][ng]
    x = torch.empty((ng.numel(), 512)); im = nt == 1
    x[im] = image_features[fr[im]]; x[~im] = entity_features[fr[~im]]
    ea = relation_type_features[torch.cat(rel_chunks)]
    er = torch.cat(rev_chunks).float().unsqueeze(1)
    if edge_mode == 'no_edge':
        el, ea, er = el[:, :0], ea[:0], er[:0]
    global_to_local[ng] = -1
    return {'x': x.to(DEVICE), 'node_type': nt.to(DEVICE), 'edge_index': el.to(DEVICE),
            'edge_attr': ea.to(DEVICE), 'edge_reverse': er.to(DEVICE), 'root_local': rl.to(DEVICE)}

@torch.no_grad()
def encode_candidates(name, image_ids, bs=128):
    spec = MODELS[name]
    comps = [image_to_component[x] for x in image_ids]
    out = [spec['model'].encode_graph(build_component_batch(comps[i:i + bs], spec['edge_mode'])).cpu()
           for i in range(0, len(comps), bs)]
    return torch.cat(out)

SPLIT = {}
def load_split(split):
    root = CLIP_ROOT / split
    cand = pd.read_parquet(root / 'candidate_images.parquet', columns=['image_id']).image_id.astype(str).tolist()
    res = pq.read_table(root / 'retrieval_results.parquet',
                        columns=['text_id', 'best_positive_rank', 'topk_image_ids', 'topk_scores', 'positive_image_id']).to_pandas()
    ti = pd.read_parquet(root / 'text_embedding_index.parquet')
    trow = dict(zip(ti.text_id.astype(str), ti.embedding_row))
    tfeat = torch.load(root / 'text_embeddings.pt', map_location='cpu', weights_only=True)
    SPLIT[split] = {'cand': cand, 'row': {x: i for i, x in enumerate(cand)}, 'res': res, 'trow': trow,
                    'tfeat': tfeat, 'text': load_queries(split), 'emb': {}}
    print(split, ':', len(cand), 'ảnh,', len(res), 'truy vấn')

def split_emb(split, name):
    S = SPLIT[split]
    if name not in S['emb']:
        S['emb'][name] = encode_candidates(name, S['cand'])
    return S['emb'][name]

@torch.no_grad()
def score_row(split, name, row):
    # Trả về: ids (50), clip_scores, graph_vec, qz, fused, hạng ảnh đúng, chỉ số ảnh hạng 1
    S, spec = SPLIT[split], MODELS[name]
    q = S['tfeat'][S['trow'][str(row.text_id)]].float().unsqueeze(0).to(DEVICE)
    qz = spec['model'].encode_query(q)[0]
    ids = [str(x) for x in row.topk_image_ids]
    emb = split_emb(split, name)
    gv = (emb[[S['row'][x] for x in ids]] @ qz.cpu()).float()
    clip = torch.tensor(row.topk_scores, dtype=torch.float32)
    fused = fuse(clip, gv, spec['alpha'])
    pos = str(row.positive_image_id)
    rank = 1 + int((fused > fused[ids.index(pos)]).sum()) if pos in ids else int(row.best_positive_rank)
    clip_rank = (ids.index(pos) + 1) if pos in ids else int(row.best_positive_rank)
    return ids, clip, gv, qz, fused, rank, clip_rank, int(torch.argmax(fused))

load_split('val'); load_split('test')
""")

md("## V1. Tái lập kết quả test của seed 2026 từ checkpoint")
code(r"""
summary = json.loads((RESULTS_ROOT / 'final_test_summary.json').read_text())
expected = summary['runs']['main_seed_2026']['test_metrics']['MRR']
if RUN_FULL_REPLAY:
    t0 = time.time(); rr = 0.0
    res = SPLIT['test']['res']
    for row in res.itertuples(index=False):
        rr += 1.0 / score_row('test', 'seed_2026', row)[5]
    got = rr / len(res)
    replay = {'MRR_replay': got, 'MRR_final_test_summary': expected, 'abs_diff': abs(got - expected),
              'n_queries': len(res), 'minutes': (time.time() - t0) / 60}
    print(replay)
    assert abs(got - expected) < 2e-4, 'Không tái lập được MRR test: dừng lại kiểm tra dữ liệu và checkpoint'
else:
    replay = {'skipped': True, 'MRR_final_test_summary': expected}
(OUT / 'V1_replay_check.json').write_text(json.dumps(replay, indent=2))
""")

md("""
## V2. Chọn K\\* trên validation

Giải thích ảnh hạng 1 mà hệ thống trả về cho `N_VAL_K` truy vấn val (seed chính). D(K) = mức giảm điểm đồ thị khi bỏ top-K đơn vị;
D\\_all = mức giảm khi bỏ mọi đơn vị có Δ dương. K\\* = K nhỏ nhất trong lưới có trung vị D(K)/D\\_all ≥ 0,8 (cùng quy tắc với cấp 2, 3).
""")
code(r"""
rng = np.random.default_rng(SELECTION_SEED)
val_res = SPLIT['val']['res']
val_rows = val_res.iloc[np.sort(rng.choice(len(val_res), N_VAL_K, replace=False))]
curves = []
for row in val_rows.itertuples(index=False):
    ids, clip, gv, qz, fused, rank, crank, top = score_row('val', 'seed_2026', row)
    g = ComponentGraph(compact, image_to_component[ids[top]], image_features, entity_features, relation_type_features, DEVICE)
    c = completeness_curve(MODELS['seed_2026']['model'], g, qz, K_GRID)
    if c is not None:
        curves.append(c)
kdf = pd.DataFrame([{'K': K, 'median_completeness': float(np.median([c[K] for c in curves])),
                     'mean_completeness': float(np.mean([c[K] for c in curves])), 'n': len(curves)} for K in K_GRID])
K_STAR = int(next((r.K for r in kdf.itertuples() if r.median_completeness >= 0.8), K_GRID[-1]))
kdf['is_K_star'] = kdf.K == K_STAR
kdf.to_csv(OUT / 'V2_Kstar_val.csv', index=False)
print(kdf.to_string(index=False)); print('K* =', K_STAR)
""")

md("""
## V3. Fidelity tại K\\* trên test, 3 seed

Cùng `N_TEST` truy vấn test (bốc một lần với `SELECTION_SEED`) cho cả 3 seed. Mỗi truy vấn: giải thích ảnh hạng 1 mà seed đó trả về,
kiểm tra điểm tái tạo khớp điểm dùng khi xếp hạng, rồi đo mọi biến thể.
""")
code(r"""
test_res = SPLIT['test']['res']
test_rows = test_res.iloc[np.sort(np.random.default_rng(SELECTION_SEED).choice(len(test_res), N_TEST, replace=False))]
rows_out = []
for seed in SEEDS:
    name = f'seed_{seed}'
    cache = OUT / f'_V3_{name}.csv'
    if cache.exists():
        rows_out.append(pd.read_csv(cache)); print(name, ': dùng lại', cache.name); continue
    spec, recs, t0 = MODELS[name], [], time.time()
    for row in test_rows.itertuples(index=False):
        ids, clip, gv, qz, fused, rank, crank, top = score_row('test', name, row)
        g = ComponentGraph(compact, image_to_component[ids[top]], image_features, entity_features, relation_type_features, DEVICE)
        s_check = float(graph_scores(spec['model'], g, qz, torch.ones((1, g.n_edges), dtype=torch.bool))[0])
        r = explain_and_measure(spec['model'], g, qz, clip, gv, spec['alpha'], top, K_STAR, n_random=N_RANDOM,
                                seed=SELECTION_SEED, query_tokens=tokens(SPLIT['test']['text'].get(str(row.text_id), '')),
                                node_label=node_label)
        r.update({'seed': seed, 'text_id': str(row.text_id), 'returned_image': ids[top],
                  'returned_is_relevant': ids[top] == str(row.positive_image_id), 'rank_positive': rank,
                  'clip_rank_positive': crank, 'replay_abs_diff': abs(s_check - float(gv[top])),
                  'S_keys': json.dumps([list(g.unit_keys[u]) for u in r['S']])})
        r.pop('S'); r['deltas_S'] = json.dumps(r['deltas_S'])
        recs.append(r)
    df = pd.DataFrame(recs); df.to_csv(cache, index=False); rows_out.append(df)
    print(f"{name}: {len(df)} truy vấn, {df.explained.sum()} có giải thích, "
          f"sai khác điểm tái tạo lớn nhất {df.replay_abs_diff.max():.2e}, {(time.time()-t0)/60:.1f} phút")
per_query = pd.concat(rows_out, ignore_index=True)
assert per_query.replay_abs_diff.max() < 1e-4, 'Điểm tái tạo không khớp điểm xếp hạng'
per_query.to_csv(OUT / 'V3_fidelity_per_query.csv', index=False)
""")

code(r"""
from scipy.stats import wilcoxon
M = ['fid_plus', 'fid_plus_rand', 'fid_plus_attn', 'fid_minus', 'fid_minus_rand', 'characterization', 'fid_perturb',
     'dscore_rm', 'dscore_rand', 'drank_rm', 'drank_rand', 'drank_attn', 'PN', 'PN_rand', 'PN_attn', 'PN_perturb',
     'PS', 'PS_rand', 'sparsity', 'size_S', 'n_units', 'in_receptive_field', 'share_contains', 'win_rate',
     'ground_S', 'ground_rand', 'ground_base', 'returned_is_relevant']
ex = per_query[per_query.explained]
per_seed = ex.groupby('seed')[M].mean()
per_seed['n'] = ex.groupby('seed').size()
per_seed['fid_minus_median'] = ex.groupby('seed').fid_minus.median()
per_seed['fid_perturb_median'] = ex.groupby('seed').fid_perturb.median()
summ = pd.DataFrame({'mean': per_seed.mean(), 'std': per_seed.std(ddof=1)})
summ.to_csv(OUT / 'V3_fidelity_summary.csv'); per_seed.to_csv(OUT / 'V3_fidelity_per_seed.csv')
print(summ.round(4).to_string())

def boot_ci(d, n=5000, seed=0):
    r = np.random.default_rng(seed); d = np.asarray(d)
    m = [r.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
avg = ex.groupby('text_id')[M].mean()
tests = []
for a, b, label in [('fid_plus', 'fid_plus_rand', 'Fid+: S so với K ngẫu nhiên'), ('fid_plus', 'fid_plus_attn', 'Fid+: S so với K attention'),
                    ('PN', 'PN_rand', 'PN: S so với K ngẫu nhiên'), ('PS', 'PS_rand', 'PS: S so với K ngẫu nhiên'),
                    ('ground_S', 'ground_rand', 'Khớp nội dung: S so với K ngẫu nhiên')]:
    d = (avg[a] - avg[b]).dropna()
    lo, hi = boot_ci(d)
    p = wilcoxon(d, alternative='greater').pvalue if (d != 0).any() else float('nan')
    tests.append({'test': label, 'n': len(d), 'mean_a': avg[a].mean(), 'mean_b': avg[b].mean(), 'delta': d.mean(),
                  'ci95_low': lo, 'ci95_high': hi, 'wilcoxon_p_one_sided': p})
tests = pd.DataFrame(tests); tests.to_csv(OUT / 'V3_tests.csv', index=False)
print(tests.round(4).to_string(index=False))
""")

md("## V4. Đường cong theo K (seed chính, cùng truy vấn test)")
code(r"""
spec, recs = MODELS['seed_2026'], []
for row in test_rows.itertuples(index=False):
    ids, clip, gv, qz, fused, rank, crank, top = score_row('test', 'seed_2026', row)
    g = ComponentGraph(compact, image_to_component[ids[top]], image_features, entity_features, relation_type_features, DEVICE)
    for K in K_GRID:
        r = explain_and_measure(spec['model'], g, qz, clip, gv, spec['alpha'], top, K, n_random=3, seed=SELECTION_SEED)
        if r['explained']:
            recs.append({'K': K, 'fid_plus': r['fid_plus'], 'fid_plus_rand': r['fid_plus_rand'], 'fid_minus': r['fid_minus'],
                         'PN': r['PN'], 'PS': r['PS'], 'sparsity': r['sparsity'], 'size_S': r['size_S']})
kc = pd.DataFrame(recs).groupby('K').mean().reset_index()
kc['is_K_star'] = kc.K == K_STAR
kc.to_csv(OUT / 'V4_sparsity_K_curve.csv', index=False); print(kc.round(4).to_string(index=False))
""")

md("## V5. Validity và ổn định giữa seed")
code(r"""
val_rows_out = []
for seed, d in ex.groupby('seed'):
    val_rows_out.append({'seed': seed, 'n': len(d), 'PN': d.PN.mean(), 'PN_rand': d.PN_rand.mean(), 'PN_attn': d.PN_attn.mean(),
                         'PN_perturb': d.PN_perturb.mean(), 'PS': d.PS.mean(), 'PS_rand': d.PS_rand.mean(),
                         'in_own_scene_graph': 1.0, 'in_receptive_field': d.in_receptive_field.mean(),
                         'ground_S': d.ground_S.mean(), 'ground_rand': d.ground_rand.mean(), 'ground_base': d.ground_base.mean(),
                         'returned_is_relevant': d.returned_is_relevant.mean()})
# Ổn định: Jaccard của S giữa các cặp seed khi cùng trả về một ảnh
jac = []
for tid, d in ex.groupby('text_id'):
    if len(d) == 3 and d.returned_image.nunique() == 1:
        sets = [set(map(tuple, json.loads(s))) for s in d.S_keys]
        pairs = [(0, 1), (0, 2), (1, 2)]
        jac.append(np.mean([len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for a, b in pairs]))
vdf = pd.DataFrame(val_rows_out)
vdf['stability_jaccard'] = float(np.mean(jac)) if jac else float('nan')
vdf['stability_n'] = len(jac)
vdf.to_csv(OUT / 'V5_validity.csv', index=False); print(vdf.round(4).to_string(index=False))

row = {'Cấp': 1, 'Dữ liệu': 'Visual Genome', 'Mô hình': 'CLIP + EdgeAwareGAT', 'Bộ giải thích': 'che quan hệ (occlusion)',
       'Số truy vấn mỗi seed': int(per_seed.n.mean()), 'K*': K_STAR}
for col, key in [('Fid+', 'fid_plus'), ('Fid+ ngẫu nhiên', 'fid_plus_rand'), ('Fid+ attention', 'fid_plus_attn'),
                 ('Characterization', 'characterization'), ('PN', 'PN'), ('PN ngẫu nhiên', 'PN_rand'), ('PS', 'PS'),
                 ('Sparsity', 'sparsity')]:
    row[col] = f"{summ.loc[key, 'mean']:.3f} ± {summ.loc[key, 'std']:.3f}"
row['Fid− (trung vị)'] = f"{summ.loc['fid_minus_median', 'mean']:.3f} ± {summ.loc['fid_minus_median', 'std']:.3f}"
row['Fid làm nhiễu (trung vị)'] = f"{summ.loc['fid_perturb_median', 'mean']:.3f} ± {summ.loc['fid_perturb_median', 'std']:.3f}"
row['Validity cấu trúc'] = f"{100 * vdf.in_receptive_field.mean():.0f}%"
row['Ổn định giữa seed (Jaccard)'] = f"{vdf.stability_jaccard.iloc[0]:.3f} (n = {len(jac)})"
pd.DataFrame([row]).to_csv(OUT / 'XAI_L3_row.csv', index=False)
print(pd.DataFrame([row]).T)
""")

md("""
## V6. 5 truy vấn thành công và 5 truy vấn thất bại (seed 2026)

Luật cố định, không nhìn giải thích: lấy `N_CASE_POOL` truy vấn test (bốc với `SELECTION_SEED`), tính hạng ảnh đúng của CLIP, mô hình
chính, `no_edge` và `random_edges`.
- **Thành công:** CLIP xếp ảnh đúng ở hạng > 1, CLIP + GAT xếp hạng 1, và cả `no_edge`, `random_edges` đều xếp hạng > 1
  (để thành công quy được cho cấu trúc quan hệ thật). Sắp theo mức tăng hạng, lấy 5.
- **Thất bại:** 3 ca "đồ thị làm hại" (CLIP hạng 1, CLIP + GAT hạng > 1) và 2 ca "không cứu được" (CLIP > 1, CLIP + GAT ở hạng 2 đến 5).
  Bốc ngẫu nhiên với `SELECTION_SEED`. Mỗi ảnh đúng chỉ xuất hiện một lần.
""")
code(r"""
pool = test_res.iloc[np.sort(np.random.default_rng(SELECTION_SEED + 1).choice(len(test_res), N_CASE_POOL, replace=False))]
rk = []
for row in pool.itertuples(index=False):
    ids, clip, gv, qz, fused, rank, crank, top = score_row('test', 'seed_2026', row)
    if str(row.positive_image_id) not in ids:
        continue
    r_no = score_row('test', 'no_edge', row)[5]
    r_rd = score_row('test', 'random_edges', row)[5]
    rk.append({'text_id': str(row.text_id), 'positive': str(row.positive_image_id), 'r_clip': crank, 'r_gat': rank,
               'r_no_edge': r_no, 'r_random': r_rd, 'top': ids[top]})
rk = pd.DataFrame(rk)
rk.to_csv(OUT / 'V6_case_pool_ranks.csv', index=False)
cnt = {'CLIP sai → GAT đúng': int(((rk.r_clip > 1) & (rk.r_gat == 1)).sum()),
       '  trong đó no_edge cũng đúng': int(((rk.r_clip > 1) & (rk.r_gat == 1) & (rk.r_no_edge == 1)).sum()),
       '  trong đó cần quan hệ thật': int(((rk.r_clip > 1) & (rk.r_gat == 1) & (rk.r_no_edge > 1) & (rk.r_random > 1)).sum()),
       'CLIP đúng → GAT sai': int(((rk.r_clip == 1) & (rk.r_gat > 1)).sum()), 'Số truy vấn': len(rk)}
print(json.dumps(cnt, ensure_ascii=False, indent=1))
(OUT / 'V6_rescue_counts.json').write_text(json.dumps(cnt, ensure_ascii=False, indent=1))

def pick(df, n, sort=None, seed=SELECTION_SEED):
    df = df.drop_duplicates('positive')
    df = df.sort_values(sort, ascending=False) if sort else df.sample(frac=1.0, random_state=seed)
    return df.head(n)
succ = pick(rk[(rk.r_clip > 1) & (rk.r_gat == 1) & (rk.r_no_edge > 1) & (rk.r_random > 1)].assign(gain=lambda d: d.r_clip - d.r_gat), 5, 'gain')
harm = pick(rk[(rk.r_clip == 1) & (rk.r_gat > 1) & ~rk.positive.isin(succ.positive)], 3)
miss = pick(rk[(rk.r_clip > 1) & (rk.r_gat.between(2, 5)) & ~rk.positive.isin(pd.concat([succ, harm]).positive)], 2)
CASES = ([('thanh_cong', r) for r in succ.itertuples(index=False)] + [('that_bai_lam_hai', r) for r in harm.itertuples(index=False)]
         + [('that_bai_khong_cuu', r) for r in miss.itertuples(index=False)])
print(len(CASES), 'ca')
""")

md("## Hình giải thích (nhúng nguyên văn `vg_xai_plot.py`): ảnh | đồ thị giải thích | bảng quan hệ, chỉ vẽ đúng tập S")
code(PLOT)
code(r"""
import requests, io
from PIL import Image

_IMG_CACHE = {}
def fetch_image(image_id):
    # Ưu tiên URL trong images.parquet, sau đó thử hai thư mục ảnh công khai của Visual Genome.
    key = str(image_id)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    urls = ([IMG_URL[key]] if IMG_URL.get(key) else []) + vg_image_urls(key)
    img = None
    for url in urls:
        try:
            r = requests.get(url, timeout=15); r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert('RGB'); break
        except Exception:
            continue
    _IMG_CACHE[key] = img
    return img

def explain_image(name, row, image_id):
    spec = MODELS[name]
    ids, clip, gv, qz, fused, rank, crank, top = score_row('test', name, row)
    i = ids.index(image_id)
    g = ComponentGraph(compact, image_to_component[image_id], image_features, entity_features, relation_type_features, DEVICE)
    s_full, s_none, deltas = unit_deltas(spec['model'], g, qz)
    S = explanation(deltas, K_STAR)
    s_rm = float(graph_scores(spec['model'], g, qz, g.keep_mask_without(S).unsqueeze(0))[0]) if S else s_full
    _, r_rm = prob_rank(clip, gv, spec['alpha'], i, s_rm)
    _, r0 = prob_rank(clip, gv, spec['alpha'], i)
    return g, S, deltas, {'S_graph': s_full, 'structure': s_full - s_none, 'rank': r0, 'rank_without_S': r_rm}

md_lines = [f'# Visual Genome: 5 truy vấn thành công và 5 truy vấn thất bại (seed 2026, K* = {K_STAR})', '']
for kind, c in CASES:
    row = test_res[test_res.text_id.astype(str) == c.text_id].iloc[0]
    text = SPLIT['test']['text'].get(c.text_id, '')
    targets = [('Ảnh đúng', c.positive)] + ([('Ảnh hạng 1 (sai)', c.top)] if c.top != c.positive else [])
    md_lines += [f'## {kind}: {c.text_id}', f'**Truy vấn:** {text}  ',
                 f'**Hạng ảnh đúng:** CLIP {c.r_clip} → CLIP + GAT {c.r_gat} · no_edge {c.r_no_edge} · random_edges {c.r_random}  ',
                 f'**Hình:** `cases/{kind}_{c.text_id}.png`', '']
    panels, first = [], None
    for lab, img_id in targets:
        g, S, deltas, info = explain_image('seed_2026', row, img_id)
        first = first or (g, S, deltas)
        panels.append((f"{lab}\nhạng {info['rank']}; bỏ giải thích S: hạng {info['rank_without_S']}", fetch_image(img_id)))
        md_lines += [f'**{lab}** ({img_id}): điểm đồ thị {info["S_graph"]:.4f}, phần do quan hệ {info["structure"]:+.4f}, '
                     f'hạng {info["rank"]}, bỏ giải thích → hạng {info["rank_without_S"]}', '',
                     '| # | Quan hệ | Δ điểm đồ thị |', '|---|---|---:|']
        md_lines += [f'| {j+1} | {unit_sentence(g, u, node_label, REL_NAME)} | {float(deltas[u]):+.2e} |' for j, u in enumerate(S)] + ['']
    # như cấp 2: đồ thị và cột Δ là giải thích của ảnh đúng
    fig = case_figure(panels, *first, node_label, REL_NAME, K_STAR,
                      f"[{kind}] {text}\nCLIP hạng {c.r_clip} → CLIP + GAT hạng {c.r_gat}")
    fig.savefig(OUT / 'cases' / f'{kind}_{c.text_id}.png', dpi=200, bbox_inches='tight'); plt.close(fig)
(OUT / 'V6_cases.md').write_text('\n'.join(md_lines), encoding='utf-8')
print('\n'.join(md_lines[:40]))
""")

md("""
## V7. Demo: nhập câu truy vấn → top-k ảnh + đồ thị giải thích

Câu tự do được mã hóa bằng CLIP ViT-B/32 (OpenAI) đóng băng; ô đầu kiểm tra encoder khớp embedding câu đã trích của nhóm
(cosine ≥ 0,99) trước khi dùng. Ứng viên là 10.443 ảnh test; CLIP lấy top-50, mô hình seed 2026 xếp hạng lại với α đã khóa.
""")
code(r"""
import open_clip
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=DEVICE)
tokenizer = open_clip.get_tokenizer('ViT-B-32'); clip_model.eval()
@torch.no_grad()
def encode_text(texts):
    return F.normalize(clip_model.encode_text(tokenizer(texts).to(DEVICE)).float(), dim=1).cpu()
chk = test_res.head(20)
mine = encode_text([SPLIT['test']['text'][str(t)] for t in chk.text_id])
theirs = F.normalize(torch.stack([SPLIT['test']['tfeat'][SPLIT['test']['trow'][str(t)]].float() for t in chk.text_id]), dim=1)
cos = (mine * theirs).sum(1)
print('cosine encoder demo so với embedding của nhóm: min', float(cos.min()))
DEMO_OK = float(cos.min()) >= 0.99
img_emb = torch.load(CLIP_ROOT / 'test/image_embeddings.pt', map_location='cpu', weights_only=True).float()
img_idx = pd.read_parquet(CLIP_ROOT / 'test/image_embedding_index.parquet')
img_ids = img_idx.sort_values('embedding_row').image_id.astype(str).tolist() if 'embedding_row' in img_idx else img_idx.image_id.astype(str).tolist()

@torch.no_grad()
def demo(text, k=3, name='seed_2026'):
    assert DEMO_OK, 'Encoder demo không khớp embedding của nhóm: không dùng demo để báo cáo'
    spec = MODELS[name]
    q = encode_text([text])[0]
    clip_all = F.normalize(img_emb, dim=1) @ q
    top50 = torch.topk(clip_all, 50).indices.tolist()
    ids = [img_ids[j] for j in top50]
    qz = spec['model'].encode_query(q.unsqueeze(0).to(DEVICE))[0]
    gv = (split_emb('test', name)[[SPLIT['test']['row'][x] for x in ids]] @ qz.cpu()).float()
    fused = fuse(clip_all[top50], gv, spec['alpha'])
    order = torch.argsort(fused, descending=True).tolist()[:k]
    fig, axes = new_figure(k)
    for c, j in enumerate(order):
        g = ComponentGraph(compact, image_to_component[ids[j]], image_features, entity_features, relation_type_features, DEVICE)
        _, _, deltas = unit_deltas(spec['model'], g, qz)
        S = explanation(deltas, K_STAR)
        header = (f"#{c+1} (CLIP hạng {j+1}), điểm cuối {float(fused[j]):+.2f}\n"
                  f"cosine CLIP {float(clip_all[top50[j]]):.3f}, điểm đồ thị {float(gv[j]):.3f}")
        plot_explained_row(axes[c], g, S, deltas, fetch_image(ids[j]), header, node_label, REL_NAME, K_STAR)
    fig.suptitle(f'Truy vấn: "{text}"  (α_CLIP = {spec["alpha"]}, K* = {K_STAR})', fontsize=9)
    fig.tight_layout()
    return fig

for q in ['a man riding a horse on the beach', 'a cat sleeping on a laptop']:
    f = demo(q); f.savefig(OUT / f"demo_{q.replace(' ', '_')[:40]}.png", dpi=200, bbox_inches='tight'); plt.show()
""")

code(r"""
info = {'notebook': '11_visualgenome_gat_xai_level1', 'K_star': K_STAR, 'seeds': SEEDS, 'main_seed': MAIN_SEED,
        'alpha_clip': {k: v['alpha'] for k, v in MODELS.items()}, 'checkpoint_sha256': {k: v['sha256'] for k, v in MODELS.items()},
        'n_val_K': N_VAL_K, 'n_test': N_TEST, 'n_random': N_RANDOM, 'K_grid': K_GRID, 'n_case_pool': N_CASE_POOL,
        'selection_seed': SELECTION_SEED, 'entity_label_column': ENT_LABEL_COL, 'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
(OUT / '11_run_info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False)); print(info)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = HERE / "11_visualgenome_gat_xai_level1.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã ghi", out)
