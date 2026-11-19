assert vg_config_ready
import time as vg_time

VG_REQUIRED_TREES = [
    "dataset/visual_genome/processed",
    "dataset/visual_genome/graph/split_v2",
    "dataset/visual_genome/clip_baseline/split_v2",
    "dataset/visual_genome/model/gat_v1/cache",
    "experiment_v2/data",
    "experiment_v2/src",
    "experiment_v2/config",
]

def vg_download_tree(remote_root, retries=4):
    local_root = VG_LOCAL_ROOT / remote_root
    local_root.mkdir(parents=True, exist_ok=True)
    entries = vg_drive.find(remote_root, detail=True)
    if isinstance(entries, list):
        entries = {path: vg_drive.info(path) for path in entries}
    files = [(path, info) for path, info in entries.items()
             if info.get("type") != "directory"]
    print(f"{remote_root}: {len(files)} files")
    for index, (remote_file, info) in enumerate(files, start=1):
        relative = Path(remote_file).relative_to(remote_root)
        local_file = local_root / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        remote_size = info.get("size")
        if local_file.exists() and remote_size is not None and local_file.stat().st_size == remote_size:
            continue
        partial = local_file.with_suffix(local_file.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                vg_drive.get_file(remote_file, str(partial))
                if remote_size is not None and partial.stat().st_size != remote_size:
                    raise IOError("size mismatch")
                partial.replace(local_file)
                break
            except Exception:
                if attempt == retries:
                    raise
                vg_time.sleep(3 * attempt)
        if index % 50 == 0 or index == len(files):
            print(f"  {index}/{len(files)}")

for vg_tree in VG_REQUIRED_TREES:
    if not vg_drive.exists(vg_tree):
        raise FileNotFoundError(f"Missing on Drive: {vg_tree}")
    vg_download_tree(vg_tree)

vg_data_ready = True
