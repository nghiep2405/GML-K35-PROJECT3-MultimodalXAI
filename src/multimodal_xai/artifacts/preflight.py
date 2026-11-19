"""Preflight known artifact layouts without importing ML dependencies."""
from pathlib import Path


def missing_inputs(pipeline, root, action="evaluate"):
    root = Path(root)
    paths = []
    if pipeline == "flickr30k":
        splits = {"prepare": ("train", "val", "test"), "train": ("train", "val"),
                  "evaluate": ("test",), "explain": ("val", "test")}[action]
        prefix = "data/Flickr30k/processed/clip_vit_b32_openai"
        for split in splits:
            for modality in ("images", "texts"):
                directory = f"{prefix}/{modality}/{split}"
                if not any((root / directory).glob("shard_*.pt")):
                    paths.append(root / directory / "shard_*.pt")
        if action == "train":
            paths.extend(_missing(root, ["data/Flickr30k/retrieval/train_hard_negatives.pt"]))
    elif pipeline == "visual_genome":
        paths = _missing(root, [
            "dataset/visual_genome/processed/splits_v2.parquet",
            "dataset/visual_genome/graph/split_v2/graph_manifest.json",
            "dataset/visual_genome/model/gat_v1/cache/compact_graph_cache_v2.pt",
        ])
        if action == "train":
            paths.extend(_missing(root, ["experiment_v2/data/full_query_pool/handoff_manifest.json"]))
    elif pipeline == "db15k":
        paths = _missing(root, ["processed/db15k/experimental_clean_v1/training_interface.json"])
    elif pipeline == "fb15k237":
        paths = _missing(root, ["processed/fb15k237_img/retrieval_bridge/manifest.json"])
    elif pipeline == "viheritage":
        manifest = "raw/collect_manifest.json" if action == "prepare" else "processed/split_manifest.json"
        paths = _missing(root, [manifest])
    return [str(path) for path in paths]


def _missing(root, paths):
    return [root / name for name in paths if not (root / name).is_file()]
