"""Resolve historical run layouts without modifying their checkpoints."""
from pathlib import Path


def flickr_checkpoint(runs: Path, reference: Path, name: str) -> Path:
    group = "3SEED" if name.startswith("seed_") else "ABLATION"
    candidates = (
        runs / name / "best_model.pt",
        reference / "runs" / name / "best_model.pt",
        reference / group / name / "best_model.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint {name!r} missing; checked {candidates}")
