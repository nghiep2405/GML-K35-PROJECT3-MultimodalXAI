"""Train CLIP + spatial GAT reranking on Flickr30k Entities."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data

from ..artifacts.atomic import atomic_json, atomic_save
from ..retrieval.ranking import ranking_metrics
from .data import OUTPUT as RETRIEVAL_ROOT, load_split
from .graph import EDGE_DIM, graph_for as build_graph, relation_type
from .model import SpatialGAT


PROJECT_ROOT = Path(os.environ.get("GRAPHGAT_PROJECT_ROOT", Path.cwd()))
RUNS_ROOT = Path(os.environ.get("FLICKR_RUNS_ROOT", PROJECT_ROOT / "results/Flickr30k/runs"))
EDGE_MODE = os.environ.get("FLICKR_EDGE_MODE", "real_edges")
SEED = int(os.environ.get("FLICKR_SEED", "2026"))
LEARNING_RATE = float(os.environ.get("FLICKR_LEARNING_RATE", "3e-4"))
RUN_SUFFIX = os.environ.get("FLICKR_RUN_SUFFIX", "").strip()
RUN_NAME_OVERRIDE = os.environ.get("FLICKR_RUN_NAME", "").strip()
MAX_EPOCHS = int(os.environ.get("FLICKR_MAX_EPOCHS", "500"))
MIN_EPOCHS = int(os.environ.get("FLICKR_MIN_EPOCHS", "100"))
PATIENCE = int(os.environ.get("FLICKR_PATIENCE", "50"))
MIN_DELTA = float(os.environ.get("FLICKR_MIN_DELTA", "0.0001"))
QUERY_BATCH = int(os.environ.get("FLICKR_QUERY_BATCH", "32"))
SMOKE_BATCHES = int(os.environ.get("FLICKR_SMOKE_BATCHES", "0"))
CHECKPOINT_EVERY = int(os.environ.get("FLICKR_CHECKPOINT_EVERY", "200"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EDGE_DIM = 8
ALPHA_GRID = [value / 10 for value in range(11)]

if EDGE_MODE not in {"real_edges", "no_edge", "random_edges"}:
    raise ValueError(f"Unsupported edge mode: {EDGE_MODE}")

RUN_NAME = RUN_NAME_OVERRIDE or f"{EDGE_MODE}_seed_{SEED}"
if RUN_SUFFIX and not RUN_NAME_OVERRIDE:
    RUN_NAME += f"_{RUN_SUFFIX.lstrip('_')}"
if SMOKE_BATCHES:
    RUN_NAME += "_smoke"
RUN_ROOT = RUNS_ROOT / RUN_NAME


def graph_for(nodes: torch.Tensor, metadata: dict[str, object], image_index: int) -> Data:
    """Compatibility entry point for the XAI notebook."""
    return build_graph(nodes, metadata, image_index, edge_mode=EDGE_MODE, seed=SEED)


def encode_candidates(model: SpatialGAT, graphs: list[Data], batch_size: int = 256) -> torch.Tensor:
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size]).to(DEVICE)
            values.append(model.encode_graph(batch).cpu())
    return torch.cat(values)


def validate(model: SpatialGAT, val: dict[str, object], val_graphs: list[Data]) -> dict[str, object]:
    graph_images = encode_candidates(model, val_graphs).to(DEVICE)
    texts = val["text_embeddings"].float().to(DEVICE)
    model.eval()
    with torch.inference_mode():
        graph_texts = model.encode_query(texts)
        clip_scores = texts @ val["global_image_embeddings"].float().to(DEVICE).T
        graph_scores = graph_texts.float() @ graph_images.float().T
        targets = val["targets"].to(DEVICE)
        clip_metrics = ranking_metrics(clip_scores, targets)
        trials = []
        for alpha in ALPHA_GRID:
            metrics = ranking_metrics((1 - alpha) * clip_scores + alpha * graph_scores, targets)
            trials.append({"alpha": alpha, **metrics})
    best = max(trials, key=lambda row: row["MRR"])
    return {"clip_only": clip_metrics, "clip_plus_graph": best, "alpha_trials": trials}


def checkpoint_payload(
    model: SpatialGAT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    history: list[dict[str, object]],
    losses: list[float],
    best_mrr: float,
    stale_epochs: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "edge_mode": EDGE_MODE,
        "seed": SEED,
        "epoch": epoch,
        "next_batch": next_batch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "losses": losses,
        "best_mrr": best_mrr,
        "stale_epochs": stale_epochs,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": str(DEVICE), "run": RUN_NAME, "edge_mode": EDGE_MODE,
                "seed": SEED, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS,
                "patience": PATIENCE, "learning_rate": LEARNING_RATE,
                "query_batch": QUERY_BATCH, "smoke_batches": SMOKE_BATCHES,
            }
        ),
        flush=True,
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train = load_split("train")
    val = load_split("val")
    negative_cache = torch.load(RETRIEVAL_ROOT / "train_hard_negatives.pt", map_location="cpu", weights_only=False)
    assert negative_cache["caption_ids"] == train["caption_ids"]
    negatives = negative_cache["negative_indices"]
    assert negatives.shape == (148915, 50)
    train_graphs = [
        graph_for(nodes, metadata, index)
        for index, (nodes, metadata) in enumerate(zip(train["graph_nodes"], train["graph_metadata"]))
    ]
    val_graphs = [
        graph_for(nodes, metadata, index)
        for index, (nodes, metadata) in enumerate(zip(val["graph_nodes"], val["graph_metadata"]))
    ]
    model = SpatialGAT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    checkpoint_path = RUN_ROOT / "last_checkpoint.pt"
    history, start_epoch, next_batch, resume_losses = [], 0, 0, []
    best_mrr, stale_epochs = -1.0, 0
    if checkpoint_path.exists() and not SMOKE_BATCHES:
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        assert state["edge_mode"] == EDGE_MODE and state["seed"] == SEED
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch, next_batch = state["epoch"], state["next_batch"]
        resume_losses = state["losses"]
        best_mrr, stale_epochs = state["best_mrr"], state["stale_epochs"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        print(f"resuming epoch={start_epoch + 1} batch={next_batch}", flush=True)

    total_batches = math.ceil(len(train["caption_ids"]) / QUERY_BATCH)
    if SMOKE_BATCHES:
        total_batches = min(total_batches, SMOKE_BATCHES)
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_started = time.perf_counter()
        generator = torch.Generator().manual_seed(SEED + epoch)
        order = torch.randperm(len(train["caption_ids"]), generator=generator)
        if SMOKE_BATCHES:
            order = order[: SMOKE_BATCHES * QUERY_BATCH]
        losses = resume_losses if epoch == start_epoch else []
        first_batch = next_batch if epoch == start_epoch else 0
        model.train()
        for batch_index in range(first_batch, total_batches):
            query_indices = order[batch_index * QUERY_BATCH : (batch_index + 1) * QUERY_BATCH]
            candidate_rows = torch.cat(
                (train["targets"][query_indices, None], negatives[query_indices]), dim=1
            )
            unique_candidates, inverse = torch.unique(candidate_rows.flatten(), return_inverse=True)
            graph_batch = Batch.from_data_list(
                [train_graphs[int(index)] for index in unique_candidates]
            ).to(DEVICE)
            query = train["text_embeddings"][query_indices].float().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            # BF16 has FP32-like exponent range and avoids the FP16 overflow that
            # corrupted the original seed-2026 run. Keep logits/loss in FP32.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                graph_values = model.encode_graph(graph_batch)
                query_values = model.encode_query(query)
                candidate_positions = inverse.view(len(query_indices), 51).to(
                    DEVICE, non_blocking=True
                )
                candidate_values = graph_values[candidate_positions]
                logits = torch.einsum(
                    "bd,bkd->bk", query_values.float(), candidate_values.float()
                ) / 0.07
                loss = F.cross_entropy(logits, torch.zeros(len(query_indices), dtype=torch.long, device=DEVICE))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch + 1} batch={batch_index + 1}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite gradient at epoch={epoch + 1} batch={batch_index + 1}"
                )
            optimizer.step()
            losses.append(float(loss.detach()))
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == total_batches:
                elapsed = time.perf_counter() - epoch_started
                rate = (batch_index + 1 - first_batch) / max(elapsed, 1e-6)
                print(
                    f"epoch={epoch+1}/{MAX_EPOCHS} batch={batch_index+1}/{total_batches} "
                    f"loss={np.mean(losses[-25:]):.5f} batches_per_second={rate:.3f}",
                    flush=True,
                )
            if not SMOKE_BATCHES and (batch_index + 1) % CHECKPOINT_EVERY == 0:
                atomic_save(
                    checkpoint_payload(
                        model, optimizer, epoch, batch_index + 1, history, losses,
                        best_mrr, stale_epochs,
                    ),
                    checkpoint_path,
                )

        validation = validate(model, val, val_graphs)
        mrr = validation["clip_plus_graph"]["MRR"]
        improved = mrr > best_mrr + MIN_DELTA
        if improved:
            best_mrr, stale_epochs = mrr, 0
            atomic_save(
                {
                    "schema_version": 1, "edge_mode": EDGE_MODE, "seed": SEED,
                    "epoch": epoch + 1, "model": model.state_dict(), "validation": validation,
                },
                RUN_ROOT / "best_model.pt",
            )
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "stale_epochs": stale_epochs,
            **validation,
        }
        history.append(record)
        atomic_json(history, RUN_ROOT / "history.json")
        atomic_save(
            checkpoint_payload(
                model, optimizer, epoch + 1, 0, history, [], best_mrr, stale_epochs
            ),
            checkpoint_path,
        )
        print(
            f"EPOCH_FINISHED epoch={epoch+1} loss={record['train_loss']:.6f} "
            f"mrr={mrr:.6f} alpha={validation['clip_plus_graph']['alpha']:.1f} "
            f"seconds={record['elapsed_seconds']:.1f} best={best_mrr:.6f}",
            flush=True,
        )
        next_batch, resume_losses = 0, []
        if SMOKE_BATCHES:
            print("SMOKE_PASS", flush=True)
            break
        if epoch + 1 >= MIN_EPOCHS and stale_epochs >= PATIENCE:
            print(f"EARLY_STOP epoch={epoch+1} stale_epochs={stale_epochs}", flush=True)
            break
    print("TRAINING_COMPLETE", RUN_ROOT, flush=True)


if __name__ == "__main__":
    main()
