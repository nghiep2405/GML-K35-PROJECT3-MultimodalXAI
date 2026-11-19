test = gat.load_split("test")
val = gat.load_split("val")
assert len(test["image_ids"]) == 1000 and len(test["caption_ids"]) == 5000
assert len(val["image_ids"]) == 1000 and len(val["caption_ids"]) == 5000
assert not set(test["image_ids"]) & set(val["image_ids"]), "test và val trùng ảnh"

MODELS, ALPHA, CKPT_META = {}, {}, {}
for name, (sub, edge_mode, seed) in CKPT_DIRS.items():
    path = FINAL / "results/Flickr30k" / sub / "best_model.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["edge_mode"] == edge_mode and int(ckpt["seed"]) == seed, f"{name}: sai edge_mode/seed"
    model = gat.SpatialGAT().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    MODELS[name] = model
    ALPHA[name] = float(ckpt["validation"]["clip_plus_graph"]["alpha"])
    CKPT_META[name] = {
        "path": str(path.relative_to(FINAL)), "sha256": sha256(path), "edge_mode": edge_mode, "seed": seed,
        "epoch": int(ckpt["epoch"]), "alpha_graph_from_validation": ALPHA[name],
        "validation_mrr": float(ckpt["validation"]["clip_plus_graph"]["MRR"]),
    }

CONFIG = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "Flickr30k Entities (test 1000 ảnh / 5000 truy vấn; val 1000 / 5000)",
    "main_run": MAIN_RUN,
    "score_formula": "S = (1 - alpha_graph) * CLIP + alpha_graph * Graph (raw cosine, như evaluate_test.py)",
    "checkpoints": CKPT_META,
    "source_sha256": {
        name: sha256(Path(gat.__file__).with_name(name))
        for name in ("train.py", "data.py", "evaluate.py", "graph.py", "model.py")
    },
    "versions": {"torch": torch.__version__, "torch_geometric": __import__("torch_geometric").__version__},
    "selection_seed": SELECTION_SEED,
}
(OUT / "B1_xai_config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(CKPT_META).T[["edge_mode", "seed", "epoch", "alpha_graph_from_validation", "validation_mrr"]]
