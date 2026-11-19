import os
from pathlib import Path
import sys
import json

configured_root = os.environ.get('GRAPHGAT_PROJECT_ROOT')
if configured_root:
    PROJECT_ROOT = Path(configured_root).expanduser().resolve()
else:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/MLGs - Project')
VG_ROOT = PROJECT_ROOT / 'dataset/visual_genome'
EXP_ROOT = PROJECT_ROOT / 'experiment_v2'
assert VG_ROOT.is_dir(), f'Missing existing Visual Genome: {VG_ROOT}'
assert EXP_ROOT.is_dir(), 'Upload the COMPLETE GML_Experiment_v2 package as experiment_v2 to Drive first'
from multimodal_xai.visual_genome.guard import WriteGuard, verify_frozen_val_test, SPLIT_SHA256
GUARD=WriteGuard(VG_ROOT,EXP_ROOT)
GRAPH_ROOT = VG_ROOT / 'graph/split_v2'
CLIP_ROOT = VG_ROOT / 'clip_baseline/split_v2'
PROCESSED_ROOT = VG_ROOT / 'processed'
print('Immutable input:', VG_ROOT)
print('Isolated output:', EXP_ROOT)
