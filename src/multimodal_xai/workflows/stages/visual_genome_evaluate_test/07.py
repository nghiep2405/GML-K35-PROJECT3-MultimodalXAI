from multimodal_xai.visual_genome.guard import verify_frozen_val_test
verify_frozen_val_test(VG_ROOT, full_sha=True)
import torch
import json
RUN_TEST_EVALUATION = os.environ.get("GRAPHGAT_ENABLE_HELDOUT_TEST", "0") == "1"  # Change True only ONCE after final model/alpha freeze
SEED=int(os.environ.get('GRAPHGAT_EVAL_SEED', '2026'))
RUN_NAME=os.environ.get('GRAPHGAT_EVAL_RUN_NAME', f'vg_full_main_seed{SEED}')  # change together with SEED for another run
OUTPUT_ROOT = Path(os.environ.get('GRAPHGAT_OUTPUT_ROOT', EXP_ROOT)).resolve()
MODEL_ROOT=OUTPUT_ROOT/'model/gat_dynamic_v2'
BEST=MODEL_ROOT/RUN_NAME/f'seed_{SEED}'/'best_gat.pt'
if BEST.is_file():
    checkpoint=torch.load(BEST,map_location='cpu',weights_only=False)
    assert checkpoint.get('architecture')=='GATConv'
    assert checkpoint.get('split_version')=='split_v2'
    assert checkpoint.get('v2_pool_sha256') is not None
    frozen_alpha=float(checkpoint['alpha'])
    ALPHA_GRID=[frozen_alpha]
    print('Validation-selected frozen alpha:',frozen_alpha)
else:
    print('Trained model checkpoint not found at:', BEST)
    print('Run 02_gat_dynamic_training_v2.ipynb first. Set RUN_TEST_EVALUATION=True only after model and alpha freeze.')
