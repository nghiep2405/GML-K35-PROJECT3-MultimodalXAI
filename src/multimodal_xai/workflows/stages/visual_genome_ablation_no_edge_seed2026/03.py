assert vg_drive_ready
from pathlib import Path
import os

VG_EDGE_MODE = "no_edge"
VG_SEED = 2026
VG_RUN_NAME = "vg_full_no_edge_seed2026"
VG_EPOCHS = 100
VG_RESUME_EPOCH = 0  # 0 = latest; e.g. 17 = resume checkpoint_epoch_017.pt
VG_LOCAL_ROOT = Path(os.environ.get("VG_LOCAL_ROOT", "/marimo/GraphGAT"))
VG_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

print("Run:", VG_RUN_NAME)
print("Local root:", VG_LOCAL_ROOT)
vg_config_ready = True
