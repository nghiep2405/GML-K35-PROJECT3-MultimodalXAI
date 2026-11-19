import matplotlib.pyplot as plt
try:
    import networkx as nx
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "networkx"], check=True); import networkx as nx
from PIL import Image

_IMG_ROOT = {}
def _img_bases(D):
    return [D.canonical_run, PHASE2, PHASE2 / "sources", PHASE2 / "sources/MKGformer/MKG/dataset", PHASE2 / "raw",
            D.canonical_run.parent, PROJECT_ROOT / "dataset", PROJECT_ROOT]

def _resolve(D, p):
    """Tìm (cột, thư mục gốc, số phần đầu cần bỏ) để đường dẫn trong bảng ảnh trỏ được tới file trên Drive."""
    parts = Path(str(p).replace("\\", "/")).parts
    parts = parts[1:] if parts and parts[0] == "/" else parts
    for base in _img_bases(D):
        for k in range(len(parts)):
            if (base / Path(*parts[k:])).is_file():
                return base, k
    return None

# FB15K-237: pipeline dữ liệu (notebook 10_fb15k237_raw_image_mapping, WP2) KHÔNG giải nén ảnh lên Drive —
# ảnh canonical chỉ nằm trong 8 file `fb15k237_image_retrieval/checkpoints/wp2_image_shards/images_shard_XXX_of_008.tar`,
# tên member trong TAR = cột image_path. Lập chỉ mục (tên file → shard, offset, size) một lần, lưu vào D.out/_cache,
# rồi đọc từng ảnh bằng seek + read đúng số byte (không giải nén cả shard, không chép ảnh lẻ lên Drive).
import tarfile, shutil
_TAR = {}
TAR_LOCAL = Path("/content/_img_tar") if Path("/content").is_dir() else Path("/tmp/_img_tar")

def _tar_shards(D):
    found = []
    for base in [D.canonical_run.parent.parent, PHASE2 / "fb15k237_image_retrieval", PHASE2 / "db15k_image_retrieval"]:
        d = base / "checkpoints" / "wp2_image_shards"
        if d.is_dir():
            found = sorted(d.glob("images_shard_*_of_*.tar"))
            if found:
                break
    return found

def _tar_index(D):
    if D.name in _TAR:
        return _TAR[D.name]
    _TAR[D.name] = None
    shards = _tar_shards(D)
    if not shards:
        return None
    cache = D.out / "_cache" / "image_tar_index.parquet"
    sig = [f"{s.name}:{s.stat().st_size}" for s in shards]
    idx = None
    if cache.exists():
        idx = pd.read_parquet(cache)
        if sorted(set(idx.shard_sig)) != sorted(sig):
            idx = None
    if idx is None:
        rows, t0 = [], time.time()
        TAR_LOCAL.mkdir(parents=True, exist_ok=True)
        for s, sg in zip(shards, sig):
            local = TAR_LOCAL / s.name
            if not (local.exists() and local.stat().st_size == s.stat().st_size):
                print(f"  chép {s.name} ({s.stat().st_size / 1e6:.0f} MB) về máy Colab để lập chỉ mục…", flush=True)
                shutil.copyfile(s, local)
            with tarfile.open(local, "r") as tf:
                for m in tf:
                    if m.isfile():
                        rows.append((Path(m.name).name, m.name, s.name, sg, m.offset_data, m.size))
        idx = pd.DataFrame(rows, columns=["base", "member", "shard", "shard_sig", "offset", "size"])
        cache.parent.mkdir(parents=True, exist_ok=True); idx.to_parquet(cache, index=False)
        print(f"{D.name}: đã lập chỉ mục {len(idx):,} ảnh trong {len(shards)} TAR ({time.time() - t0:.0f}s) → {cache}")
    by_base = {b: (sh, int(o), int(z)) for b, sh, o, z in zip(idx["base"], idx["shard"], idx["offset"], idx["size"])}
    _TAR[D.name] = (shards[0].parent, by_base)
    return _TAR[D.name]

def _open_from_tar(D, p):
    hit = _tar_index(D)
    if not hit or not isinstance(p, str):
        return None
    folder, by_base = hit; rec = by_base.get(Path(p.replace("\\", "/")).name)
    if rec is None:
        return None
    shard, off, size = rec
    src = TAR_LOCAL / shard if (TAR_LOCAL / shard).exists() else folder / shard
    with open(src, "rb") as f:
        f.seek(off); data = f.read(size)
    return Image.open(io.BytesIO(data)).convert("RGB")

def open_image(D, row):
    if not getattr(D, "img_paths", None):
        return None
    if D.name not in _IMG_ROOT:                     # dò một lần trên vài ảnh đầu, rồi dùng cho mọi ảnh
        _IMG_ROOT[D.name] = None
        for ci, col in enumerate(D.img_paths):
            probe = [col.get(int(row))] + col.dropna().head(5).tolist()
            for p in [x for x in probe if isinstance(x, str)]:
                hit = _resolve(D, p)
                if hit:
                    _IMG_ROOT[D.name] = (ci,) + hit; break
            if _IMG_ROOT[D.name]:
                break
        if not _IMG_ROOT[D.name]:                   # không có file lẻ → thử đọc trong TAR shard
            for ci, col in enumerate(D.img_paths):
                probe = [col.get(int(row))] + col.dropna().head(5).tolist()
                if any(_open_from_tar(D, x) is not None for x in probe if isinstance(x, str)):
                    _IMG_ROOT[D.name] = ("tar", ci); break
        root = _IMG_ROOT[D.name]
        print(f"{D.name}: nguồn ảnh =", (f"TAR shard trong {_TAR[D.name][0]}" if root and root[0] == "tar" else root) or
              f"KHÔNG TÌM THẤY (ví dụ đường dẫn: {[c.dropna().head(1).tolist() for c in D.img_paths]})")
    if not _IMG_ROOT[D.name]:
        return None
    try:
        if _IMG_ROOT[D.name][0] == "tar":
            return _open_from_tar(D, D.img_paths[_IMG_ROOT[D.name][1]].get(int(row)))
        ci, base, k = _IMG_ROOT[D.name]; p = D.img_paths[ci].get(int(row))
        if not isinstance(p, str):
            return None
        parts = Path(p.replace("\\", "/")).parts; parts = parts[1:] if parts and parts[0] == "/" else parts
        f = base / Path(*parts[k:]) if len(parts) > k else None
        return Image.open(f).convert("RGB") if f is not None and f.is_file() else None
    except Exception:
        return None

def explain_image(ctx, D, sc, pos):
    ex = explain_modality(ctx, D, sc, pos)
    # E3 (bộ ba train) chỉ cho truy vấn gián tiếp: điểm đồ thị = RotatE(anchor, r, e) đi qua đúng một quan hệ.
    # Truy vấn trực tiếp: DB15K chỉ dùng CLIP; FB15K dùng cos(chiếu truy vấn, joint(e)) — không đi qua quan hệ nào, và
    # ablation cấp 2 (xáo cạnh / đồ thị ngẫu nhiên vẫn giữ nguyên mức tăng nhóm trực tiếp) cho thấy bộ ba không đóng góp,
    # nên vẽ bộ ba ở đây sẽ là giải thích hậu kiểm (đề cấm). Với FB15K trực tiếp chỉ giữ E2 (đóng góp theo phương thức).
    if ctx["q"].prediction_direction != "direct":
        ex["E3"] = explain_facts(ctx, D, sc, pos, [K_STAR[D.name]])
    return ex

def draw_case(ax, D, q, ex, K):
    G = nx.DiGraph(); e = ex["entity"]; a = int(q.anchor_entity_index)
    G.add_node(a, kind="anchor"); G.add_node(e, kind="cand")
    G.add_edge(a, e, label=D.rel_label[int(q.relation_index)] + " ?", kind="query")
    facts = [f for f in ex.get("E3", {}).get("facts", [])[:K] if f["delta"] > 0]
    vmax = max([f["delta"] for f in facts] + [1e-9])
    size = {}
    for f in facts:
        for (u, v, fid) in f["path"]:
            h, t, r = (int(x) for x in D.train[fid])
            for n in (u, v):
                if n not in G:
                    G.add_node(n, kind="other")
            G.add_edge(h, t, label=D.rel_label[r], kind="fact")
        x = f["head"] if f["tail"] == e else f["tail"]; size[x] = size.get(x, 0) + f["delta"]
    pos = nx.spring_layout(G, seed=0, k=1.2)
    col = {"anchor": "#a882ff", "cand": "#f2c94c", "other": "#cfcfcf"}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col[G.nodes[n]["kind"]] for n in G],
                           node_size=[900 if G.nodes[n]["kind"] != "other" else 250 + 900 * size.get(n, 0) / vmax for n in G])
    fe = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "fact"]; qe = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "query"]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=fe, edge_color="#7a5cc4", width=1.6, arrows=True, arrowsize=12)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=qe, edge_color="#999", style="dashed", width=1.2, arrows=True)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: D.ent_label[n][:22] for n in G}, font_size=7)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={(u, v): d["label"][:24] for u, v, d in G.edges(data=True)}, font_size=6)
    ax.set_title("Bằng chứng đồ thị (E3): nét liền = bộ ba train, nét đứt = quan hệ được hỏi", fontsize=8); ax.axis("off")

def case_figure(name, case_row):
    D, sc, Sp = DATA[name], SCORERS[(name, MAIN_SEED)], TEST[name]
    c = Sp.ctx(int(case_row.qi), sc); order = top_order(c)
    tpos = [int(p) for p in order.tolist() if bool(c["tmask"][p])]
    out = {}
    if tpos:
        out["dung"] = (tpos[0], explain_image(c, D, sc, tpos[0]))
    if int(order[0]) not in tpos:
        out["hang1"] = (int(order[0]), explain_image(c, D, sc, int(order[0])))
    n_img = len(out); fig, axes = plt.subplots(1, n_img + 2, figsize=(4.2 * (n_img + 2), 4.4))
    for ax, (tag, (p, ex)) in zip(axes, out.items()):
        img = open_image(D, int(c["rows"][p]))
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(.5, .5, "ảnh không có trên Drive", ha="center", va="center")
        ax.set_title(f"{'Ảnh đúng' if tag == 'dung' else 'Ảnh hạng 1 (sai)'}: {ex['label'][:28]}\nhạng {ex['rank']}", fontsize=8); ax.axis("off")
    key = "dung" if "dung" in out else "hang1"; ex = out[key][1]
    draw_case(axes[n_img], D, c["q"], ex, K_STAR[name])
    ax = axes[n_img + 1]; labels, vals = [], []
    for m, d in ex["drop"].items():
        labels.append(f"{MOD[m]}\nw={ex['weights'][m]:.2f}"); vals.append(d["delta"])
    ax.barh(labels, vals, color="#a882ff"); ax.axvline(0, color="#555", lw=.8)
    ax.set_title("E2: điểm đồ thị giảm khi che phương thức", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(f"[{case_row.case}] {c['q'].query_text}\nCLIP hạng {case_row.clip_rank} → CLIP + AdaMF hạng {case_row.hybrid_rank}", fontsize=9)
    fig.tight_layout(); f = D.out / "cases" / f"{case_row.case}_{case_row.query_id}.png"; fig.savefig(f, dpi=130); plt.close(fig)
    return out, f

def describe(D, q, ex, K):
    lines = [f"- Thực thể quyết định (E1): **{ex['label']}** · điểm đồ thị {ex['g']:+.3f} · hạng {ex['rank']}"]
    w = ", ".join(f"{MOD[m]} {ex['weights'][m]:.2f}" for m in range(3) if ex["present"][m])
    dr = ", ".join(f"{MOD[m]} Δ {d['delta']:+.3f} (hạng → {d['rank']})" for m, d in ex["drop"].items())
    lines.append(f"- Phương thức (E2): attention {w}; che từng phương thức: {dr or 'chỉ có một phương thức'}")
    E3 = ex.get("E3")
    if E3 and E3["ok"]:
        lines.append(f"- Bộ ba train của thực thể: {E3['n_facts']} · điểm gốc {E3['s_orig']:+.3f} · bản học lại đủ bộ ba {E3['s_full']:+.3f}")
        lines.append("\n| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |\n|---|---|---|---:|")
        for j, f in enumerate([f for f in E3["facts"] if f["delta"] > 0][:K], 1):
            path = " → ".join([D.ent_label[f["path"][0][0]]] + [D.ent_label[v] for (_, v, _) in f["path"]])
            hop = f"{len(f['path'])} bước" if f["path"][0][0] == int(q.anchor_entity_index) else "lân cận (không nối về anchor)"
            lines.append(f"| {j} | {f['text']} | {path} ({hop}) | {f['delta']:+.4f} |")
        top = E3["removal"].get(f"top@{K}"); rnd = E3["removal"].get(f"rand@{K}")
        if top:
            lines.append(f"\nBỏ đồng thời top-{K}: điểm giảm {top['dscore']:+.4f}, hạng thay đổi {top['drank']:+.1f}"
                         + (f" · bỏ {K} bộ ba ngẫu nhiên (trung bình {N_RANDOM} lần): {rnd['dscore']:+.4f}, {rnd['drank']:+.1f}" if rnd else ""))
    elif E3 is not None:
        lines.append("- Thực thể không có bộ ba train nào → điểm đồ thị chỉ đến từ ảnh/văn bản (cold-start).")
    return "\n".join(lines)

CASE_MD = {}
for name in DATASET_NAMES:
    D = DATA[name]; md = [f"# {name} — 5 truy vấn thành công và 5 truy vấn thất bại (seed {MAIN_SEED}, K = {K_STAR[name]})\n"]
    for _, row in CASES[name].iterrows():
        out, fpath = case_figure(name, row)
        md.append(f"\n## {row.case} — {row.query_id}\n**Truy vấn:** {row.query_text}  \n**Hạng ảnh đúng:** CLIP {row.clip_rank} → "
                  f"CLIP + AdaMF {row.hybrid_rank}  \n**Hình:** `cases/{fpath.name}`\n")
        for tag, (p, ex) in out.items():
            md.append(f"\n**{'Ảnh đúng' if tag == 'dung' else 'Ảnh xếp hạng 1 (sai)'}**\n" + describe(D, TEST[name].q.iloc[int(row.qi)], ex, K_STAR[name]))
    CASE_MD[name] = "\n".join(md); (D.out / "C8_case_analysis.md").write_text(CASE_MD[name], encoding="utf-8")
    print(f"{name}: đã lưu {len(CASES[name])} ca vào {D.out / 'cases'}")
display(Markdown(CASE_MD[DATASET_NAMES[0]][:6000]))
