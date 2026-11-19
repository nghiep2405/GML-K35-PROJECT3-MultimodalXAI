REL_NAMES = {0: "tự vòng", 1: "ảnh→vùng (chứa)", 2: "vùng→ảnh (thuộc)", 3: "chồng lấn",
             4: "nằm bên trái", 5: "nằm bên phải", 6: "nằm phía trên", 7: "nằm phía dưới"}
REL_SHORT = {0: "self", 1: "contains", 2: "part_of", 3: "overlaps", 4: "left_of", 5: "right_of",
             6: "above", 7: "below"}

def relation_groups(data):
    """Gom cạnh (trừ tự vòng) thành các quan hệ vô hướng {u, v}."""
    src, dst = data.edge_index
    types = data.edge_attr.argmax(1)
    groups = {}
    for e, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        groups.setdefault(key, {"edges": [], "types": {}})
        groups[key]["edges"].append(e)
        groups[key]["types"][(u, v)] = int(types[e])
    out = []
    for (u, v), g in sorted(groups.items()):
        kind = "star" if u == 0 else "spatial"
        out.append({"nodes": (u, v), "edges": g["edges"], "types": g["types"], "kind": kind})
    return out

def scores_with_masks(model, data, q_proj, masks, chunk=256):
    """Điểm đồ thị q'·z của MỘT ảnh dưới nhiều mặt nạ cạnh (mỗi hàng của masks là một mặt nạ)."""
    out = []
    x, ei, ea = data.x.cpu(), data.edge_index.cpu(), data.edge_attr.cpu()   # đồ thị gốc luôn ở CPU
    masks = masks.cpu()
    for start in range(0, len(masks), chunk):
        datas = [Data(x=x, edge_index=ei[:, m], edge_attr=ea[m]) for m in masks[start:start + chunk]]
        batch = Batch.from_data_list(datas).to(DEVICE)
        z = model.encode_graph(batch).float()
        out.append((z @ q_proj.to(DEVICE)).cpu())
    return torch.cat(out)

def encode_with_attention(model, data):
    """Chạy lại đúng encode_graph nhưng trả thêm attention của 2 tầng GAT."""
    d = data.clone().to(DEVICE)   # clone: Data.to() của PyG đổi thiết bị tại chỗ, không được làm hỏng đồ thị gốc
    h1, (ei1, a1) = model.gat1(d.x, d.edge_index, d.edge_attr, return_attention_weights=True)
    enc, (ei2, a2) = model.gat2(F.elu(h1), d.edge_index, d.edge_attr, return_attention_weights=True)
    z = F.normalize(model.graph_norm(d.x[0:1] + enc[0:1]), dim=-1)
    return z.squeeze(0).cpu(), a1.mean(1).cpu(), a2.mean(1).cpu()

# --- Kiểm tra tính đúng của lõi XAI ---
model0, graphs0 = MODELS[MAIN_RUN], test_graphs_for(MAIN_RUN)
rng = np.random.default_rng(SELECTION_SEED)
check_q = rng.choice(len(test["caption_ids"]), 50, replace=False)
max_err_full, max_err_att, changed = 0.0, 0.0, 0
for qi in check_q:
    img = int(T_TARGET[qi])
    data = graphs0[img]
    E = data.edge_index.shape[1]
    full = torch.ones(E, dtype=torch.bool)
    selfonly = data.edge_index[0] == data.edge_index[1]
    s = scores_with_masks(model0, data, QPROJ_TEST[MAIN_RUN][qi], torch.stack([full, selfonly]))
    max_err_full = max(max_err_full, abs(float(s[0]) - float(GRAPH_TEST[MAIN_RUN][qi, img])))
    z_att, _, _ = encode_with_attention(model0, data)
    max_err_att = max(max_err_att, float((z_att - GEMB_TEST[MAIN_RUN][img]).abs().max()))
    changed += int(abs(float(s[0] - s[1])) > 1e-6)
print(f"B4 | mặt nạ đầy đủ = điểm gốc: sai số tối đa {max_err_full:.2e}")
print(f"B4 | encode_with_attention = encode_graph: sai số tối đa {max_err_att:.2e}")
print(f"B4 | bỏ hết quan hệ làm điểm thay đổi: {changed}/{len(check_q)} truy vấn")
assert max_err_full < 1e-4 and max_err_att < 1e-4 and changed > 0
(OUT / "B4_core_checks.json").write_text(json.dumps(
    {"max_err_full_mask": max_err_full, "max_err_attention_path": max_err_att,
     "queries_changed_when_all_relations_removed": changed, "n_checked": len(check_q)}, indent=2))
