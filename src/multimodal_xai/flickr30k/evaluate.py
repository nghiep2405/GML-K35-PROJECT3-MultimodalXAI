"""Evaluate locked Flickr30k checkpoints on test without test-time tuning.

The best epoch and graph fusion weight are read from each checkpoint's stored
validation result. The test split is never used for model or alpha selection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from . import train as gat
from ..artifacts.layouts import flickr_checkpoint


PROJECT_ROOT = Path(os.environ.get("GRAPHGAT_PROJECT_ROOT", Path.cwd()))
OUTPUT = Path(os.environ.get("FLICKR_SUMMARY", "test-result/runs/flickr30k/final_test_summary.json"))
RUN_CONFIGS = (
    {"name": "seed_2026", "edge_mode": "real_edges", "seed": 2026},
    {"name": "seed_2027", "edge_mode": "real_edges", "seed": 2027},
    {"name": "seed_2028", "edge_mode": "real_edges", "seed": 2028},
    {"name": "no_edge", "edge_mode": "no_edge", "seed": 2026},
    {"name": "random_edges", "edge_mode": "random_edges", "seed": 2026},
)
METRICS = ("Recall@1", "Recall@5", "Recall@10", "Recall@50", "MRR")


def evaluate(
    config: dict[str, object],
    test: dict[str, object],
    graph_cache: dict[tuple[str, int], list[gat.Data]],
) -> dict[str, object]:
    reference = Path(os.environ.get("FLICKR_REFERENCE_ROOT", PROJECT_ROOT / "results/Flickr30k"))
    checkpoint_path = flickr_checkpoint(gat.RUNS_ROOT, reference, str(config["name"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("edge_mode") != config["edge_mode"]:
        raise ValueError(f"edge_mode mismatch in {checkpoint_path}")
    if int(checkpoint.get("seed", -1)) != config["seed"]:
        raise ValueError(f"seed mismatch in {checkpoint_path}")
    validation = checkpoint.get("validation")
    if not validation or "clip_plus_graph" not in validation:
        raise ValueError(f"validation selection metadata missing in {checkpoint_path}")
    alpha_graph = float(validation["clip_plus_graph"]["alpha"])

    edge_mode = str(config["edge_mode"])
    graph_key = (edge_mode, int(config["seed"]))
    if graph_key not in graph_cache:
        graph_cache[graph_key] = [
            gat.build_graph(nodes, metadata, index, edge_mode=edge_mode, seed=graph_key[1])
            for index, (nodes, metadata) in enumerate(
                zip(test["graph_nodes"], test["graph_metadata"])
            )
        ]

    model = gat.SpatialGAT().to(gat.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    graph_images = gat.encode_candidates(model, graph_cache[graph_key]).to(gat.DEVICE)
    texts = test["text_embeddings"].float().to(gat.DEVICE)
    images = test["global_image_embeddings"].float().to(gat.DEVICE)
    targets = test["targets"].to(gat.DEVICE)
    with torch.inference_mode():
        graph_texts = model.encode_query(texts)
        clip_scores = texts @ images.T
        graph_scores = graph_texts.float() @ graph_images.float().T
        combined = (1.0 - alpha_graph) * clip_scores + alpha_graph * graph_scores
        clip_metrics = gat.ranking_metrics(clip_scores, targets)
        combined_metrics = gat.ranking_metrics(combined, targets)
    del model, graph_images, texts, images, targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run": config["name"],
        "edge_mode": edge_mode,
        "seed": config["seed"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "alpha_graph_locked_from_validation": alpha_graph,
        "alpha_clip": 1.0 - alpha_graph,
        "validation_metrics_at_checkpoint": validation["clip_plus_graph"],
        "test_clip_only": clip_metrics,
        "test_clip_plus_graph": combined_metrics,
    }


def main() -> None:
    test = gat.load_split("test")
    if len(test["image_ids"]) != 1000 or len(test["caption_ids"]) != 5000:
        raise ValueError("unexpected Flickr30k test split size")
    graph_cache: dict[tuple[str, int], list[gat.Data]] = {}
    results = [evaluate(config, test, graph_cache) for config in RUN_CONFIGS]
    main_results = [result for result in results if result["edge_mode"] == "real_edges"]
    aggregate = {}
    for metric in METRICS:
        values = np.asarray(
            [result["test_clip_plus_graph"][metric] for result in main_results],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    summary = {
        "schema_version": 1,
        "dataset": "Flickr30k Entities",
        "split": "test",
        "selection_policy": "best checkpoint and graph alpha selected on validation; no test tuning",
        "score_formula": "(1-alpha_graph)*clip_score + alpha_graph*graph_score",
        "candidate_images": len(test["image_ids"]),
        "test_queries": len(test["caption_ids"]),
        "runs": results,
        "main_three_seed_mean_std": aggregate,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT)
    print(OUTPUT)
    for metric, values in aggregate.items():
        print(f"{metric}: {values['mean']:.6f} ± {values['std_sample']:.6f}")


if __name__ == "__main__":
    main()
