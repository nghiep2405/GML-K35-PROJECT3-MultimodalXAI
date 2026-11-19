def explain(model, data, q_proj, alpha, K, labels=None):
    groups = relation_groups(data)
    E = data.edge_index.shape[1]
    is_self = data.edge_index[0] == data.edge_index[1]
    masks = [torch.ones(E, dtype=torch.bool), is_self.clone()]
    for g in groups:
        m = torch.ones(E, dtype=torch.bool)
        m[g["edges"]] = False
        masks.append(m)
    s = scores_with_masks(model, data, q_proj, torch.stack(masks))
    s_full, s_none = float(s[0]), float(s[1])
    deltas = (s[0] - s[2:]).tolist()
    order = sorted(range(len(groups)), key=lambda j: -deltas[j])
    chosen = [j for j in order if deltas[j] > 0][:K]
    keep = is_self.clone()
    delete = torch.ones(E, dtype=torch.bool)
    for j in chosen:
        keep[groups[j]["edges"]] = True
        delete[groups[j]["edges"]] = False
    delete_all_pos = torch.ones(E, dtype=torch.bool)
    for j, d in enumerate(deltas):
        if d > 0:
            delete_all_pos[groups[j]["edges"]] = False
    s_keep, s_del, s_del_allpos = [float(v) for v in scores_with_masks(
        model, data, q_proj, torch.stack([keep, delete, delete_all_pos]))]
    _, att1, att2 = encode_with_attention(model, data)
    lab = labels or {}
    def name(n):
        return "ẢNH" if n == 0 else lab.get(n, f"R{n - 1}")
    rels = []
    for rank, j in enumerate(order):
        g = groups[j]
        u, v = g["nodes"]
        if g["kind"] == "star":
            text = f"{name(v)} –[thuộc]→ ẢNH"
            path = f"{name(v)} → ẢNH"
        else:
            t_uv = g["types"].get((u, v), g["types"].get((v, u)))
            text = f"{name(u)} –[{REL_NAMES[t_uv]}]→ {name(v)}"
            path = f"{name(u)} ↔ {name(v)} → ẢNH (2 bước)"
        rels.append({
            "rank": rank + 1, "kind": g["kind"], "nodes": [u, v], "relation": text, "path": path,
            "type_ids": {f"{a}->{b}": t for (a, b), t in g["types"].items()},
            "delta_graph": deltas[j], "delta_fused": alpha * deltas[j],
            "attention_l1": float(att1[g["edges"]].sum()), "attention_l2": float(att2[g["edges"]].sum()),
            "in_explanation": j in chosen,
        })
    n_groups = len(groups)
    struct = s_full - s_none
    return {
        "S_full": s_full, "S_none": s_none, "structure_contribution": struct,
        "S_keep_explanation": s_keep, "S_delete_explanation": s_del, "S_delete_all_positive": s_del_allpos,
        "comprehensiveness": (s_full - s_del) / (s_full - s_del_allpos) if s_full - s_del_allpos > 1e-8 else float("nan"),
        "sufficiency": (s_keep - s_none) / struct if abs(struct) > 1e-8 else float("nan"),
        "n_relations": n_groups, "n_explanation": len(chosen),
        "sparsity": 1 - len(chosen) / n_groups if n_groups else float("nan"),
        "n_spatial_in_explanation": sum(groups[j]["kind"] == "spatial" for j in chosen),
        "top_spatial": next((r for r in rels if r["kind"] == "spatial"), None),
        "relations": rels, "_groups": groups, "_chosen": chosen, "_deltas": deltas,
    }

# Ví dụ nhanh trên 1 truy vấn test
_qi = int(check_q[0]); _img = int(T_TARGET[_qi])
_ex = explain(MODELS[MAIN_RUN], graphs0[_img], QPROJ_TEST[MAIN_RUN][_qi], ALPHA[MAIN_RUN], K=5)
print(test["captions"][_qi])
print({k: round(v, 4) if isinstance(v, float) else v for k, v in _ex.items() if not k.startswith("_") and k != "relations"})
pd.DataFrame(_ex["relations"][:8])[["rank", "relation", "delta_graph", "attention_l2", "in_explanation"]]
