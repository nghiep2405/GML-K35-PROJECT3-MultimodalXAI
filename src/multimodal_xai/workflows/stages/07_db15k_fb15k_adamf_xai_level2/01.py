# C0. Cấu hình
import os, sys, json, glob, math, random, time, hashlib, subprocess, io, base64, re, inspect
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
for pkg in ["pyarrow", "transformers"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from IPython.display import HTML, display, Markdown

def find_project_root():
    env = os.environ.get("XAI2_PROJECT_ROOT")
    if env:
        return Path(env)
    pats = ["/content/drive/MyDrive/MLGs - Project", "/content/drive/MyDrive/*/MLGs - Project",
            "/content/drive/Shareddrives/*/MLGs - Project", "/content/drive/MyDrive/.shortcut-targets-by-id/*/MLGs - Project"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError("Không thấy thư mục 'MLGs - Project'. Thêm shortcut vào My Drive hoặc đặt XAI2_PROJECT_ROOT.")

PROJECT_ROOT = find_project_root()
PHASE2 = Path(os.environ.get("XAI2_DATA_ROOT", PROJECT_ROOT / "dataset" / "adamf")).resolve()
FINAL = Path(os.environ.get("XAI2_FINAL_ROOT", PROJECT_ROOT / "0. final"))
SEEDS = [2027, 2028, 2029]
MAIN_SEED = 2027                       # seed dùng cho ca phân tích và demo (thứ tự đầu tiên, không chọn theo kết quả)
DATASET_NAMES = [x for x in os.environ.get("XAI2_DATASETS", "DB15K,FB15K").split(",") if x]
SELECTION_SEED = 2026
TOPK = 1000                            # đúng top-k rerank của nhóm
N_BENCH_DIRECT, N_BENCH_HEAD, N_BENCH_TAIL = 10, 5, 5
M_CAND_FACTS = int(os.environ.get("XAI2_M", "24"))   # số bộ ba ứng viên được đo riêng cho mỗi giải thích
K_GRID = [1, 2, 3, 5, 8]
COMPREHENSIVENESS_TARGET = 0.80
N_VAL_K = int(os.environ.get("XAI2_NVAL", "40"))     # số truy vấn validation để chọn K
N_FID = int(os.environ.get("XAI2_NFID", "60"))       # số truy vấn test mỗi seed để kiểm chứng
N_RANDOM = 10                                        # số lần bỏ ngẫu nhiên cho mỗi giải thích
MIMIC = dict(steps=int(os.environ.get("XAI2_STEPS", "300")), lr=0.01, n_neg=16, max_facts=int(os.environ.get("XAI2_MAXF", "256")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
WRITE_TO_DRIVE = os.environ.get("XAI2_WRITE_DRIVE", "0") == "1"
REUSE_DONE = os.environ.get("XAI2_REUSE", "0") == "1"   # True: C6/C9 đã có file kết quả thì đọc lại thay vì tính lại (tính lại cho kết quả y hệt vì tất định)

def out_dir(ds):
    base = (FINAL / "results" / ds / "XAI") if (WRITE_TO_DRIVE and FINAL.exists()) else (Path(os.environ.get("XAI2_OUT", "/tmp/xai2_out")) / ds)
    for sub in ["", "cases", "figures"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base

print("PROJECT_ROOT:", PROJECT_ROOT, "| DEVICE:", DEVICE, "| datasets:", DATASET_NAMES)
