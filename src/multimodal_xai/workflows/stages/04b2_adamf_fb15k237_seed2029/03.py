from pathlib import Path
import sys, subprocess, json, os, hashlib, shutil, time
import torch
if not os.environ.get('ADAMF_PROJECT_ROOT'):
    from google.colab import drive
    drive.mount('/content/drive')
assert torch.cuda.is_available(), 'Bật GPU trong Runtime → Change runtime type.'
PROJECT_ROOT = Path(os.environ.get('ADAMF_PROJECT_ROOT', '/content/drive/MyDrive/MLGs - Project')).expanduser().resolve()
PHASE2_ROOT = Path(os.environ.get('ADAMF_DATA_ROOT', PROJECT_ROOT / 'dataset' / 'adamf')).resolve()
OUTPUT_ROOT = Path(os.environ.get('ADAMF_RUN_ROOT', PHASE2_ROOT / 'model' / 'adamf_v1')).expanduser().resolve()
RUNS = OUTPUT_ROOT
NOTEBOOK_DIR = PROJECT_ROOT / 'notebook'
REPO = Path(os.environ.get('ADAMF_REPO', '/content/AdaMF-MAT-fb15k237-retrieval'))
RUN_NAME = 'fb15k237_clean_kgc_to_retrieval_50ep_seed2029_v1'
RESUME_MODE = os.environ.get("ADAMF_RESUME", "1")
REPO_REF = 'main'
SESSION_TRAIN_HOURS = 8.0
CHECKPOINT_MINUTES = 30.0
# Test split chỉ được đánh giá ở ô cuối sau khi hoàn tất huấn luyện.
SEEDS = [2029]
RUN = RUNS / RUN_NAME
RESUME_TRAINING = (RUN / "identity.json").is_file() if RESUME_MODE == "auto" else RESUME_MODE == "1"
assert PROJECT_ROOT.is_dir() and PHASE2_ROOT.is_dir(), 'Kiểm tra shortcut Drive và PROJECT_ROOT.'
assert SESSION_TRAIN_HOURS > 0 and CHECKPOINT_MINUTES > 0
RUNS.mkdir(parents=True,exist_ok=True)
if RESUME_TRAINING:
    assert (RUN/'identity.json').is_file(), 'Không thấy run cũ để resume.'
else:
    existing=[p for seed in SEEDS for p in [RUN/f'seed_{seed}'/'last_checkpoint.pt',RUN/f'seed_{seed}'/'previous_checkpoint.pt',RUN/f'seed_{seed}'/'complete.json'] if p.exists()]
    assert not existing, f'Run đã có checkpoint: {existing}. Đổi RESUME_TRAINING=True hoặc dùng RUN_NAME mới.'
print('Python:',sys.version,'\nTorch:',torch.__version__,'\nGPU:',torch.cuda.get_device_name(0))
print('DATA ROOT:',PHASE2_ROOT)
for seed in SEEDS:
    print('CHECKPOINT:', RUN / f'seed_{seed}' / 'last_checkpoint.pt')
print('NOTEBOOK FOLDER:',NOTEBOOK_DIR)

CFG=dict(epochs=50,seeds=SEEDS,dim=64,learning_rate=1e-4,weight_decay=1e-4,
         max_train_queries=417700,train_negatives=50,query_batch_size=8,
         hard_negative_pool=200,rerank_topk=1000,validation_queries=520370,
         checkpoint_every_batches=500,selection_seed=2026,freeze_features=True,
         alpha_clip_grid=[round(i/10,1) for i in range(11)],
         normalization='per_query_zscore_sample_std',mu=0.0,
         early_stopping_patience=5,early_stopping_min_delta=1e-5)
CACHE_ROOT=OUTPUT_ROOT/'fb15k237_retrieval_cache_v2'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
RUN.mkdir(parents=True,exist_ok=True)
LOCAL_CACHE=Path(os.environ.get('ADAMF_LOCAL_CACHE', OUTPUT_ROOT / 'work/cache'))
LOCAL_CACHE.mkdir(parents=True,exist_ok=True)
print(json.dumps(CFG,indent=2))
print('AdaMF fused/query dimension:',2*CFG['dim'],'(128 giống output GAT 04a)')
