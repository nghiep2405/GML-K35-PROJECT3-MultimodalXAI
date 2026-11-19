import torch
import pandas as pd
from pathlib import Path

emb_root = EXP_ROOT / "data/embeddings"

indices = sorted((emb_root / "indexes").glob("part_*.parquet"))

assert len(indices) == 102

total = 0

for i, index_path in enumerate(indices, 1):
    shard_path = emb_root / "shards" / f"{index_path.stem}.pt"

    assert shard_path.is_file()

    index = pd.read_parquet(index_path)

    tensor = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=True
    )

    assert tensor.ndim == 2
    assert tensor.shape == (len(index), 512)
    assert torch.isfinite(tensor).all()

    total += len(index)

    print(f"\rValidated embeddings: {i}/102", end="")

assert total == 1_670_618

print("\n✅ ALL EMBEDDING SHARDS PASSED")
