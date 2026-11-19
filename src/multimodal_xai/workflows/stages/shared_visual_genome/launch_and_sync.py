assert vg_sync_ready

def vg_run_and_sync(check_interval=30):
    import os as vg_os
    import subprocess as vg_process_module
    import time as vg_clock

    local_run = VG_LOCAL_ROOT / vg_remote_run_path()
    remote_run = vg_remote_run_path()
    environment = vg_os.environ.copy()
    environment["GRAPHGAT_PROJECT_ROOT"] = str(VG_LOCAL_ROOT)
    environment["GRAPHGAT_EDGE_MODE"] = VG_EDGE_MODE
    environment["GRAPHGAT_SEEDS"] = str(VG_SEED)
    environment["GRAPHGAT_RUN_NAME"] = VG_RUN_NAME
    environment["GRAPHGAT_QUERIES_PER_IMAGE"] = "3"
    environment["GRAPHGAT_EPOCHS"] = str(VG_EPOCHS)
    environment["GRAPHGAT_RUN_TRAINING"] = "1"
    environment["GRAPHGAT_SKIP_VALIDATION"] = "0"
    environment["GRAPHGAT_RANDOM_EDGE_SEED"] = "2026"
    environment["PYTHONUNBUFFERED"] = "1"

    process = vg_process_module.Popen(
        [vg_sys.executable, "-u", "-m", "multimodal_xai.visual_genome.train"],
        cwd=str(VG_LOCAL_ROOT), env=environment,
    )
    last_signature = None
    uploaded_epoch_snapshots = set()

    def sync_outputs(force=False):
        nonlocal last_signature
        checkpoint = local_run / "last_checkpoint.pt"
        if not checkpoint.exists():
            return
        signature = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        snapshots = sorted(local_run.glob("checkpoint_epoch_*.pt"))
        new_snapshots = [
            path for path in snapshots if path.name not in uploaded_epoch_snapshots
        ]
        if not force and signature == last_signature and not new_snapshots:
            return
        if force or signature != last_signature:
            for filename in ("history.json", "last_checkpoint.pt", "best_gat.pt"):
                local_file = local_run / filename
                if local_file.exists():
                    vg_upload_file(local_file, f"{remote_run}/{filename}")
                    print("Uploaded:", filename)
        for snapshot in new_snapshots:
            vg_upload_file(snapshot, f"{remote_run}/{snapshot.name}")
            uploaded_epoch_snapshots.add(snapshot.name)
            print("Uploaded immutable epoch snapshot:", snapshot.name)
        last_signature = signature

    try:
        while process.poll() is None:
            sync_outputs()
            vg_clock.sleep(check_interval)
        sync_outputs(force=True)
    except KeyboardInterrupt:
        process.send_signal(2)
        process.wait()
        sync_outputs(force=True)
        raise
    print("Training exit code:", process.returncode)
    if process.returncode != 0:
        raise RuntimeError(f"Training failed: {process.returncode}")

print("Starting:", VG_RUN_NAME)
vg_run_and_sync()
