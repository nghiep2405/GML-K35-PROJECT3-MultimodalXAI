assert vg_preflight_ready
import json as vg_json
import subprocess as vg_subprocess
import sys as vg_sys

def vg_remote_run_path():
    return f"experiment_v2/model/gat_dynamic_v2/{VG_RUN_NAME}/seed_{VG_SEED}"

def vg_download_resume():
    remote_run = vg_remote_run_path()
    local_run = VG_LOCAL_ROOT / remote_run
    local_run.mkdir(parents=True, exist_ok=True)
    if not vg_drive.exists(remote_run):
        print("No remote checkpoint; starting new run")
        return
    if VG_RESUME_EPOCH > 0:
        snapshot_name = f"checkpoint_epoch_{VG_RESUME_EPOCH:03d}.pt"
        source = f"{remote_run}/{snapshot_name}"
        if not vg_drive.exists(source):
            raise FileNotFoundError(f"Missing requested recovery point: {source}")
        destination = local_run / "last_checkpoint.pt"
        partial = destination.with_suffix(".pt.part")
        vg_drive.get_file(source, str(partial))
        partial.replace(destination)
        for stale_name in ("history.json", "best_gat.pt"):
            stale = local_run / stale_name
            if stale.exists():
                stale.unlink()
        print("Resuming explicitly from:", snapshot_name)
        return
    for filename in ("history.json", "last_checkpoint.pt", "best_gat.pt"):
        source = f"{remote_run}/{filename}"
        if not vg_drive.exists(source):
            continue
        destination = local_run / filename
        partial = destination.with_suffix(destination.suffix + ".part")
        vg_drive.get_file(source, str(partial))
        partial.replace(destination)
        print("Resume downloaded:", filename)

def vg_upload_file(local_file, remote_file):
    parent = str(Path(remote_file).parent)
    vg_drive.makedirs(parent, exist_ok=True)
    vg_drive.put_file(str(local_file), remote_file)

vg_download_resume()
vg_sync_ready = True
