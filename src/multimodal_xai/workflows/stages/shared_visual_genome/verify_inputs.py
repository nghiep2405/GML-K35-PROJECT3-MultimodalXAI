assert vg_data_ready
import importlib.util
import torch

assert importlib.util.find_spec("multimodal_xai.visual_genome.train")

VG_REQUIRED_FILES = [
    VG_LOCAL_ROOT / "dataset/visual_genome/processed/splits_v2.parquet",
    VG_LOCAL_ROOT / "dataset/visual_genome/graph/split_v2/graph_manifest.json",
    VG_LOCAL_ROOT / "dataset/visual_genome/model/gat_v1/cache/compact_graph_cache_v2.pt",
    VG_LOCAL_ROOT / "experiment_v2/data/full_query_pool/handoff_manifest.json",
]
for vg_required_file in VG_REQUIRED_FILES:
    print("OK" if vg_required_file.exists() else "MISSING", vg_required_file)
assert all(path.exists() for path in VG_REQUIRED_FILES)
if not torch.cuda.is_available():
    raise RuntimeError("Chọn GPU runtime rồi chạy lại notebook từ đầu")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
vg_preflight_ready = True
