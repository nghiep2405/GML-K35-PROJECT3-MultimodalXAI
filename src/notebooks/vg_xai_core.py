# Lõi bộ giải thích cấp 1 cho CLIP + EdgeAwareGAT trên Visual Genome.
# Được nhúng nguyên văn vào notebook 11_visualgenome_gat_xai_level1.ipynb (ô "Lõi bộ giải thích").
#
# Đơn vị giải thích = một quan hệ của scene graph trong ảnh: cạnh gốc và cạnh ngược của nó
# (cùng hai đầu, cùng loại quan hệ) gộp thành một đơn vị. Cạnh ảnh -> thực thể ("contains") cũng là đơn vị.
# Đóng góp của đơn vị u: delta_u = S_full - S_(full \ u), trong đó S là điểm đồ thị q'.z của ảnh và
# z được tính lại bằng chính 2 tầng GAT của checkpoint sau khi bỏ u (giống bộ giải thích của notebook 06).
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


class ComponentGraph:
    """Đồ thị của một ảnh (một thành phần liên thông trong compact cache), đặt sẵn trên thiết bị."""

    def __init__(self, compact, comp, image_features, entity_features, relation_type_features,
                 device, edge_mode="real_edges", random_edge_seed=2026):
        ns, ne = compact["node_ptr"][comp:comp + 2].tolist()
        es, ee = compact["edge_ptr"][comp:comp + 2].tolist()
        nodes = compact["node_perm"][ns:ne]
        edges = compact["edge_index"][:, es:ee].clone()
        if edge_mode == "random_edges":
            gen = torch.Generator().manual_seed(int(random_edge_seed) + int(comp))
            perm = torch.randperm(edges.shape[1], generator=gen)
            edges = torch.stack((edges[0], edges[1, perm]))
        g2l = {int(n): i for i, n in enumerate(nodes.tolist())}
        src = torch.tensor([g2l[int(v)] for v in edges[0].tolist()], dtype=torch.long)
        dst = torch.tensor([g2l[int(v)] for v in edges[1].tolist()], dtype=torch.long)
        node_type = compact["node_type"][nodes].long()
        roots = (node_type == 1).nonzero().flatten()
        assert roots.numel() == 1, "mỗi thành phần phải có đúng một nút ảnh"
        feat_rows = compact["node_feature_row"][nodes]
        x = torch.empty((nodes.numel(), image_features.shape[1]), dtype=torch.float32)
        img = node_type == 1
        x[img] = image_features[feat_rows[img]]
        x[~img] = entity_features[feat_rows[~img]]
        rel = compact["edge_relation_type"][es:ee].long()
        rev = compact["edge_is_reverse"][es:ee].float()

        self.comp = int(comp)
        self.device = device
        self.edge_mode = edge_mode
        self.nodes_global = nodes
        self.feature_rows = feat_rows
        self.node_type = node_type.to(device)
        self.x = x.to(device)
        self.root = int(roots.item())
        self.src, self.dst = src, dst
        self.rel = rel
        self.rev = rev
        self.edge_attr = relation_type_features[rel].to(device)
        self.edge_reverse = rev.unsqueeze(1).to(device)
        self.n_nodes = int(nodes.numel())
        self.n_edges = int(src.numel())
        # Gộp cạnh gốc và cạnh ngược thành đơn vị quan hệ.
        groups = {}
        for e in range(self.n_edges):
            a, b = int(src[e]), int(dst[e])
            key = (min(a, b), max(a, b), int(rel[e]))
            groups.setdefault(key, []).append(e)
        self.unit_keys = list(groups.keys())
        self.unit_edges = [groups[k] for k in self.unit_keys]
        self.n_units = len(self.unit_keys)
        um = torch.zeros((self.n_units, self.n_edges), dtype=torch.bool)
        for u, es_ in enumerate(self.unit_edges):
            um[u, es_] = True
        self.unit_edge_mask = um
        # Khoảng cách tới nút ảnh (vô hướng) để kiểm tra vùng tiếp nhận 2 tầng.
        adj = [[] for _ in range(self.n_nodes)]
        for a, b in zip(src.tolist(), dst.tolist()):
            adj[a].append(b)
            adj[b].append(a)
        dist = [math.inf] * self.n_nodes
        dist[self.root] = 0
        frontier = [self.root]
        while frontier:
            nxt = []
            for v in frontier:
                for w in adj[v]:
                    if dist[w] == math.inf:
                        dist[w] = dist[v] + 1
                        nxt.append(w)
            frontier = nxt
        self.dist = dist
        self.unit_is_contains = [k[2] == 0 for k in self.unit_keys]

    def unit_in_receptive_field(self, u):
        a, b, _ = self.unit_keys[u]
        return min(self.dist[a], self.dist[b]) <= 1

    def keep_mask_without(self, units):
        keep = torch.ones(self.n_edges, dtype=torch.bool)
        if len(units):
            keep &= ~self.unit_edge_mask[list(units)].any(0)
        return keep

    def keep_mask_only(self, units):
        if len(units) == 0:
            return torch.zeros(self.n_edges, dtype=torch.bool)
        return self.unit_edge_mask[list(units)].any(0)


def _encode_batched(model, g, masks, edge_index_override=None):
    """Mã hóa nút ảnh cho nhiều biến thể của cùng một đồ thị trong một lần chạy.

    masks: bool [M, E] (cạnh được giữ). edge_index_override: list độ dài M gồm (src, dst) thay thế hoặc None.
    """
    M = masks.shape[0]
    n = g.n_nodes
    xs, types, eis, eas, ers, roots = [], [], [], [], [], []
    for m in range(M):
        keep = masks[m]
        if edge_index_override is not None and edge_index_override[m] is not None:
            s, d = edge_index_override[m]
        else:
            s, d = g.src, g.dst
        ei = torch.stack((s[keep], d[keep])) + m * n
        eis.append(ei)
        eas.append(g.edge_attr[keep.to(g.device)])
        ers.append(g.edge_reverse[keep.to(g.device)])
        roots.append(g.root + m * n)
    x = g.x.repeat(M, 1)
    nt = g.node_type.repeat(M)
    batch = {
        "x": x,
        "node_type": nt,
        "edge_index": torch.cat(eis, 1).to(g.device),
        "edge_attr": torch.cat(eas, 0),
        "edge_reverse": torch.cat(ers, 0),
        "root_local": torch.tensor(roots, device=g.device),
    }
    if g.edge_mode == "no_edge":
        batch["edge_index"] = batch["edge_index"][:, :0]
        batch["edge_attr"] = batch["edge_attr"][:0]
        batch["edge_reverse"] = batch["edge_reverse"][:0]
    return model.encode_graph(batch)


@torch.no_grad()
def graph_scores(model, g, qz, masks, edge_index_override=None, chunk=64):
    """Điểm đồ thị q'.z của ảnh cho từng biến thể (mask)."""
    out = []
    for i in range(0, masks.shape[0], chunk):
        ov = None if edge_index_override is None else edge_index_override[i:i + chunk]
        z = _encode_batched(model, g, masks[i:i + chunk], ov)
        out.append(z @ qz)
    return torch.cat(out).float().cpu()


@torch.no_grad()
def unit_deltas(model, g, qz):
    """delta_u = S_full - S_(full \\ u) cho mọi đơn vị; trả về (S_full, S_none, delta[U])."""
    full = torch.ones((1, g.n_edges), dtype=torch.bool)
    none = torch.zeros((1, g.n_edges), dtype=torch.bool)
    leave = ~g.unit_edge_mask  # [U, E]: bỏ đúng một đơn vị
    s = graph_scores(model, g, qz, torch.cat([full, none, leave], 0))
    return float(s[0]), float(s[1]), (s[0] - s[2:])


def explanation(deltas, K):
    """S = tối đa K đơn vị có delta dương lớn nhất."""
    order = torch.argsort(deltas, descending=True).tolist()
    return [u for u in order[:K] if deltas[u] > 0]


@torch.no_grad()
def attention_unit_scores(model, g):
    """Điểm attention của từng đơn vị (tổng attention trung bình các head, cả hai tầng). Chỉ làm đường cơ sở."""
    ei = torch.stack((g.src, g.dst)).to(g.device)
    h = model.node_projection(g.x) + model.type_embedding(g.node_type)
    ea = model.edge_projection(torch.cat([g.edge_attr, g.edge_reverse], 1))
    upd, (ei1, a1) = model.gat1(h, ei, edge_attr=ea, return_attention_weights=True)
    h = model.norm1(h + F.elu(upd))
    _, (ei2, a2) = model.gat2(h, ei, edge_attr=ea, return_attention_weights=True)
    # GATConv không tự thêm self-loop (add_self_loops=False) nên thứ tự cạnh giữ nguyên.
    assert ei1.shape[1] == g.n_edges and ei2.shape[1] == g.n_edges
    per_edge = (a1.mean(1) + a2.mean(1)).cpu()
    return torch.stack([per_edge[e].sum() for e in g.unit_edges])


def rewired_edges(g, units, rng):
    """Làm nhiễu S: thay đầu không phải nút ảnh của mỗi cạnh trong S bằng một thực thể ngẫu nhiên khác của ảnh."""
    s, d = g.src.clone(), g.dst.clone()
    entities = [i for i in range(g.n_nodes) if i != g.root]
    for u in units:
        a, b, _ = g.unit_keys[u]
        for e in g.unit_edges[u]:
            # giữ đầu gần nút ảnh hơn, thay đầu còn lại
            if g.dist[int(s[e])] <= g.dist[int(d[e])]:
                cur = int(d[e])
                pool = [v for v in entities if v != cur and v != int(s[e])]
                if pool:
                    d[e] = rng.choice(pool)
            else:
                cur = int(s[e])
                pool = [v for v in entities if v != cur and v != int(d[e])]
                if pool:
                    s[e] = rng.choice(pool)
    return s, d


def fuse(clip_scores, graph_scores_vec, alpha):
    cz = (clip_scores - clip_scores.mean()) / clip_scores.std().clamp_min(1e-6)
    gz = (graph_scores_vec - graph_scores_vec.mean()) / graph_scores_vec.std().clamp_min(1e-6)
    return alpha * cz + (1.0 - alpha) * gz


def prob_rank(clip_scores, graph_vec, alpha, i, new_graph_score=None):
    """Xác suất softmax và hạng của ứng viên i khi điểm đồ thị của i được thay bằng new_graph_score."""
    gv = graph_vec.clone()
    if new_graph_score is not None:
        gv[i] = new_graph_score
    fused = fuse(clip_scores, gv, alpha)
    p = torch.softmax(fused, 0)[i].item()
    rank = 1 + int((fused > fused[i]).sum())
    return p, rank


def characterization(fid_plus, fid_minus):
    a = min(max(fid_plus, 0.0), 1.0)
    b = 1.0 - min(max(fid_minus, 0.0), 1.0)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def tokens(text):
    import re
    return set(w for w in re.findall(r"[a-z]+", str(text).lower()) if len(w) > 2)


def unit_grounded(g, u, query_tokens, node_label):
    """Đơn vị chạm một thực thể có từ xuất hiện trong câu truy vấn (validity theo nội dung)."""
    a, b, _ = g.unit_keys[u]
    for v in (a, b):
        if v == g.root:
            continue
        if tokens(node_label(g, v)) & query_tokens:
            return True
    return False


@torch.no_grad()
def explain_and_measure(model, g, qz, clip_scores, graph_vec, alpha, i, K, n_random=10, seed=2026,
                        query_tokens=None, node_label=None):
    """Đo đủ bộ chỉ số cho giải thích của ứng viên i (ảnh hạng 1 trả về)."""
    rng = random.Random(seed)
    s_full, s_none, deltas = unit_deltas(model, g, qz)
    S = explanation(deltas, K)
    U = g.n_units
    k = len(S)
    p0, r0 = prob_rank(clip_scores, graph_vec, alpha, i)
    out = {"S_full": s_full, "S_none": s_none, "n_units": U, "size_S": k, "p0": p0, "rank0": r0,
           "S": S, "deltas_S": [float(deltas[u]) for u in S]}
    if k == 0:
        out["explained"] = False
        return out
    out["explained"] = True
    masks = [g.keep_mask_without(S), g.keep_mask_only(S)]
    rand_sets = [rng.sample(range(U), k) for _ in range(n_random)]
    masks += [g.keep_mask_without(R) for R in rand_sets]
    masks += [g.keep_mask_only(R) for R in rand_sets]
    att = attention_unit_scores(model, g)
    A = torch.argsort(att, descending=True).tolist()[:k]
    masks += [g.keep_mask_without(A)]
    sc = graph_scores(model, g, qz, torch.stack(masks))
    s_rm, s_only = float(sc[0]), float(sc[1])
    s_rand_rm = sc[2:2 + n_random]
    s_rand_only = sc[2 + n_random:2 + 2 * n_random]
    s_att_rm = float(sc[-1])
    p_rm, r_rm = prob_rank(clip_scores, graph_vec, alpha, i, s_rm)
    p_only, r_only = prob_rank(clip_scores, graph_vec, alpha, i, s_only)
    p_att, r_att = prob_rank(clip_scores, graph_vec, alpha, i, s_att_rm)
    pr = [prob_rank(clip_scores, graph_vec, alpha, i, float(v)) for v in s_rand_rm]
    po = [prob_rank(clip_scores, graph_vec, alpha, i, float(v)) for v in s_rand_only]
    # làm nhiễu: thay đầu kia
    s_pert, d_pert = rewired_edges(g, S, rng)
    s_perturb = float(graph_scores(model, g, qz, torch.ones((1, g.n_edges), dtype=torch.bool),
                                   edge_index_override=[(s_pert, d_pert)])[0])
    p_pert, r_pert = prob_rank(clip_scores, graph_vec, alpha, i, s_perturb)
    fp = (p0 - p_rm) / p0
    fm = (p0 - p_only) / p0
    out.update({
        "fid_plus": fp,
        "fid_plus_rand": float(np.mean([(p0 - p) / p0 for p, _ in pr])),
        "fid_plus_attn": (p0 - p_att) / p0,
        "fid_minus": fm,
        "fid_minus_rand": float(np.mean([(p0 - p) / p0 for p, _ in po])),
        "characterization": characterization(fp, fm),
        "fid_perturb": (p0 - p_pert) / p0,
        "dscore_rm": s_full - s_rm,
        "dscore_rand": float(np.mean([s_full - float(v) for v in s_rand_rm])),
        "drank_rm": r_rm - r0,
        "drank_rand": float(np.mean([r - r0 for _, r in pr])),
        "drank_attn": r_att - r0,
        "PN": float(r_rm > 1),
        "PN_rand": float(np.mean([r > 1 for _, r in pr])),
        "PN_attn": float(r_att > 1),
        "PN_perturb": float(r_pert > 1),
        "PS": float(r_only == 1),
        "PS_rand": float(np.mean([r == 1 for _, r in po])),
        "sparsity": 1.0 - k / U,
        "in_receptive_field": float(np.mean([g.unit_in_receptive_field(u) for u in S])),
        "share_contains": float(np.mean([g.unit_is_contains[u] for u in S])),
        "win_rate": float(np.mean([(p0 - p_rm) > (p0 - p) for p, _ in pr])),
    })
    if query_tokens is not None and node_label is not None:
        out["ground_S"] = float(np.mean([unit_grounded(g, u, query_tokens, node_label) for u in S]))
        out["ground_rand"] = float(np.mean([np.mean([unit_grounded(g, u, query_tokens, node_label) for u in R])
                                            for R in rand_sets]))
        out["ground_base"] = float(np.mean([unit_grounded(g, u, query_tokens, node_label) for u in range(U)]))
    return out


@torch.no_grad()
def completeness_curve(model, g, qz, Ks):
    """D(K)/D_all: mức giảm điểm đồ thị khi bỏ top-K so với bỏ mọi đơn vị có delta dương (để chọn K* trên val)."""
    s_full, _, deltas = unit_deltas(model, g, qz)
    pos = [u for u in torch.argsort(deltas, descending=True).tolist() if deltas[u] > 0]
    if not pos:
        return None
    masks = [g.keep_mask_without(pos)] + [g.keep_mask_without(pos[:K]) for K in Ks]
    sc = graph_scores(model, g, qz, torch.stack(masks))
    d_all = s_full - float(sc[0])
    if d_all <= 1e-9:
        return None
    return {K: (s_full - float(sc[1 + j])) / d_all for j, K in enumerate(Ks)}
