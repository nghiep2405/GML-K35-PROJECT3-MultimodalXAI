from pathlib import Path
import json

EMB_ROOT = EXP_ROOT / "data/embeddings"
NEG_ROOT = EXP_ROOT / "data/hard_negatives"

shards = sorted((EMB_ROOT / "shards").glob("part_*.pt"))
indexes = sorted((EMB_ROOT / "indexes").glob("part_*.parquet"))
negatives = sorted((NEG_ROOT / "parts").glob("part_*.parquet"))

print("=== EMBEDDING STATUS ===")
print("Embedding tensors:", len(shards))
print("Embedding indexes:", len(indexes))
print("Hard-negative parts:", len(negatives))

print("\n=== MANIFEST STATUS ===")

for path in [
    EMB_ROOT / "manifest.json",
    NEG_ROOT / "manifest.json"
]:
    print(f"{path.parent.name}/manifest.json:", path.exists())

    if path.exists():
        data = json.loads(path.read_text())
        print("Queries:", data.get("queries"))
        print("Completed:", data.get("completed", "Not specified"))

print("\n=== EMBEDDING FILE STATUS ===")

if shards:
    print("First shard:", shards[0].name)
    print("Last shard:", shards[-1].name)
    print("Last shard size:", round(shards[-1].stat().st_size / 1024**2, 2), "MB")

print("\n=== CONCLUSION ===")

if len(shards) == 102 and len(indexes) == 102:
    print("✅ All 102 embedding shard/index pairs exist.")
else:
    print("⚠️ Embedding shards/indexes are incomplete.")

if len(negatives) == 102:
    print("✅ All 102 hard-negative parts exist.")
