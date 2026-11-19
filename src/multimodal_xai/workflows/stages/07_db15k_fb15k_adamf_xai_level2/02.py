# C0b. Nạp dữ liệu và checkpoint của nhóm (đúng đường dẫn trong notebook huấn luyện 04b1/04b2)
from multimodal_xai.workflows.runner import run_shared
run_shared('artifacts', globals())






def load_dataset(name):
    D = DS(); D.name = name; D.kind = "db" if name == "DB15K" else "fb"
    if D.kind == "db":
        clean = PHASE2 / "processed/db15k/experimental_clean_v1"
        retrieval = PHASE2 / "db15k_image_retrieval/runs/db15k_clean_v1"
        man = json.loads((retrieval / "manifest.json").read_text())
        art = {k: PHASE2 / v["phase2_relative_path"] for k, v in man["artifacts"].items()}
        refs = json.loads(art["canonical_references"].read_text())
        for k, v in refs["artifacts"].items():
            art.setdefault(k, PHASE2 / v["phase2_relative_path"])
        repro = clean / "reproduction"
        D.canonical_run = PHASE2 / refs["canonical_run"]
        D.runs = {s: PHASE2 / f"model/adamf_v1/db15k_clean_kgc_to_retrieval_50ep_seed{s}_v1" for s in SEEDS}
        D.ckpt = {s: D.runs[s] / "best_adamf.pt" for s in SEEDS}
        D.locked_dir = {s: D.runs[s] for s in SEEDS}
        checkpoint_root = os.environ.get("XAI2_CHECKPOINT_ROOT")
        if checkpoint_root:
            D.runs = {s: Path(checkpoint_root) / f"seed_{s}" for s in SEEDS}
            D.ckpt = {s: D.runs[s] / "best_adamf.pt" for s in SEEDS}
            D.locked_dir = D.runs.copy()
        enc = json.loads((D.runs[MAIN_SEED] / "clip_encoder.json").read_text())
        D.encoder = (enc["name"], enc["revision"])
    else:
        bridge = json.loads((PHASE2 / "processed/fb15k237_img/retrieval_bridge/manifest.json").read_text())
        art = {k: PHASE2 / v["phase2_relative_path"] for k, v in bridge["artifacts"].items()}
        repro = PHASE2 / bridge["promoted_reproduction"]
        D.canonical_run = PHASE2 / bridge["canonical_root"]
        D.runs = {s: PHASE2 / f"model/adamf_v1/fb15k237_clean_kgc_to_retrieval_50ep_seed{s}_v1" for s in SEEDS}
        D.ckpt = {s: D.runs[s] / f"seed_{s}" / "best_adamf.pt" for s in SEEDS}
        D.locked_dir = {s: D.runs[s] / f"seed_{s}" for s in SEEDS}
        checkpoint_root = os.environ.get("XAI2_CHECKPOINT_ROOT")
        if checkpoint_root:
            D.runs = {s: Path(checkpoint_root) / f"seed_{s}" for s in SEEDS}
            D.ckpt = {s: D.runs[s] / "best_adamf.pt" for s in SEEDS}
            D.locked_dir = D.runs.copy()
        enc = json.loads((D.runs[MAIN_SEED] / "text_encoder.json").read_text())
        D.encoder = (enc["model_name"], enc["revision"])
    D.ent_raw = read_mapping(repro / "entity2id.txt"); D.rel_raw = read_mapping(repro / "relation2id.txt")
    D.N, D.R = len(D.ent_raw), len(D.rel_raw)
    D.train = read_triples(repro / "train2id.txt")
    ents = pd.read_parquet(art["entities"]).set_index("entity_index").reindex(range(D.N))
    D.ent_label = ents.entity_label.fillna(pd.Series(D.ent_raw)).astype(str).to_numpy()
    D.rel_label = np.array([rel_label(r) for r in D.rel_raw])
    q = pd.read_parquet(art["queries"])
    D.test_q = q[q["split"] == "test"].sort_values("query_id").reset_index(drop=True)
    D.val_q = q[q["split"] == "val"].sort_values("query_id").reset_index(drop=True)
    rel = pd.read_parquet(art["relevance"])
    rel = rel[rel.query_id.isin(set(D.test_q.query_id) | set(D.val_q.query_id))]
    idx = pd.read_parquet(art["image_feature_index"])
    D.row_map = idx.set_index("image_id").feature_row
    rel = rel.assign(row=rel.image_id.map(D.row_map))
    D.pos = rel.groupby("query_id").row.apply(lambda x: np.unique(x.to_numpy(np.int64))).to_dict()
    pools = pd.read_parquet(art["candidate_pools"])
    D.pool = {sp: np.sort(pools.loc[pools["split"] == sp, "image_id"].map(D.row_map).to_numpy(np.int64)) for sp in ["val", "test"]}
    D.img_feat = np.load(art["clip_image_features"], mmap_mode="r", allow_pickle=False)
    links = pd.read_parquet(art["entity_image"])[["image_id", "entity_index"]].drop_duplicates()
    links["row"] = links.image_id.map(D.row_map)
    lists = links.groupby("row").entity_index.apply(list).to_dict()
    L = max(map(len, lists.values()))
    D.img_ent = torch.full((len(idx), L), -1, dtype=torch.long)
    for r, ids in lists.items():
        D.img_ent[int(r), :len(ids)] = torch.tensor(ids)
    try:                                           # mọi cột đường dẫn ảnh; open_image sẽ tự dò thư mục gốc
        ims = pd.read_parquet(art["images"]).drop_duplicates("image_id").set_index("image_id")
        order = idx.sort_values("feature_row").image_id
        D.img_paths = [pd.Series(ims[c].reindex(order).to_numpy(), index=range(len(idx)))
                       for c in ims.columns if "path" in c.lower() and (ims[c].dtype == object or pd.api.types.is_string_dtype(ims[c]))]
    except Exception as ex:
        print("Không đọc được bảng đường dẫn ảnh:", ex); D.img_paths = []
    D.img_id = idx.sort_values("feature_row").image_id.to_numpy()
    # Đồ thị train cho giải thích (chỉ bộ ba train — không có val/test)
    D.facts_of = {}
    for i, (h, t, r) in enumerate(D.train):
        D.facts_of.setdefault(int(h), []).append(i); D.facts_of.setdefault(int(t), []).append(i)
    D.neighbors = {e: set(np.where(D.train[fs, 0] == e, D.train[fs, 1], D.train[fs, 0]).tolist()) for e, fs in
                   ((e, np.array(v)) for e, v in D.facts_of.items())}
    D.out = out_dir(name)
    print(f"{name}: N={D.N} R={D.R} train={len(D.train)} test_q={len(D.test_q)} val_q={len(D.val_q)} "
          f"test_pool={len(D.pool['test'])} ảnh/thực thể tối đa={L}")
    return D

DATA = {name: load_dataset(name) for name in DATASET_NAMES}
