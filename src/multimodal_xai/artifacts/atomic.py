"""Atomic replacement of generated artifacts."""
import json
from pathlib import Path


def atomic_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    import torch

    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
