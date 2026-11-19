"""Flickr30k ranking with competition ties."""
from __future__ import annotations


def ranking_metrics(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    import torch

    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite validation scores")
    positive = scores.gather(1, targets[:, None])
    ranks = 1 + (scores > positive).sum(dim=1)
    return {
        "Recall@1": float((ranks <= 1).float().mean()),
        "Recall@5": float((ranks <= 5).float().mean()),
        "Recall@10": float((ranks <= 10).float().mean()),
        "Recall@50": float((ranks <= 50).float().mean()),
        "MRR": float((1.0 / ranks.float()).mean()),
    }
