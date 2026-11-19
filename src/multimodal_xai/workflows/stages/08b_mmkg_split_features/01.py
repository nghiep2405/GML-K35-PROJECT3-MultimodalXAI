# B0. Cấu hình và nạp đầu ra của 08a
import os, sys, re, json, glob, math, hashlib, shutil, subprocess, unicodedata, html, textwrap
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
import numpy as np, pandas as pd
from PIL import Image
from IPython.display import display

def find_project_root():
    env = os.environ.get("MMKG_PROJECT_ROOT")
    if not env and os.environ.get("MMKG_DATA_ROOT"):
        return Path.cwd()
    if env:
        return Path(env)
    for p in ["/content/drive/MyDrive/MLGs - Project", "/content/drive/MyDrive/*/MLGs - Project",
              "/content/drive/Shareddrives/*/MLGs - Project", "/content/drive/MyDrive/.shortcut-targets-by-id/*/MLGs - Project"]:
        hits = sorted(glob.glob(p))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError("Không thấy thư mục 'MLGs - Project'.")

PROJECT_ROOT = find_project_root()
FINAL = PROJECT_ROOT / "0. final"
MMKG = Path(os.environ.get("MMKG_DATA_ROOT", FINAL / "data" / "MMKG"))
RAW, PROC, IMG, REVIEW = MMKG / "raw", MMKG / "processed", MMKG / "images", MMKG / "review"
REPRO = PROC / "reproduction"
SRC = MMKG / "generated-source"
for d in [REPRO, SRC]:
    d.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 2025                 # seed chia dữ liệu, khác seed chấm ảnh (2026) và seed huấn luyện (2027–2029)
SPLIT_FRAC = (0.70, 0.15, 0.15)   # train / val / test. KG nhỏ (~430 triple) nên 15% cho val, test để có ≥ 40–60 truy vấn gián tiếp mỗi bên
SYMMETRIC = {"P47", "NEARBY"}     # hai chiều luôn chung split
NO_QUERY_REL = {"P47", "P279"}    # hai đầu đều không có ảnh → không sinh truy vấn
D1_MIN_ACC = 0.85                 # độ chính xác tối thiểu của ảnh lấy ở category con (P373_d1)
ALLOW_ONE_RATER = os.environ.get("MMKG_ONE_RATER", "0") == "1"   # chỉ có 1 người chấm: vẫn chạy, κ = không có
IMG_MODEL = "clip-ViT-B-32"
TXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
TZ = timezone(timedelta(hours=7))

ents = pd.read_csv(PROC / "entities.csv", keep_default_na=False)
rels = pd.read_csv(PROC / "relations.csv", keep_default_na=False)
trip = pd.read_csv(PROC / "triples_all.tsv", sep="\t", keep_default_na=False)
imgs = pd.read_csv(PROC / "images.csv", keep_default_na=False)
collect = json.loads((RAW / "collect_manifest.json").read_text(encoding="utf-8"))
N, R = len(ents), len(rels)
assert (ents.entity_index.to_numpy() == np.arange(N)).all() and (rels.relation_index.to_numpy() == np.arange(R)).all()
LAB = dict(zip(ents.entity_index, ents.label_vi)); LAB_EN = dict(zip(ents.entity_index, ents.label_en))
RLAB = dict(zip(rels.relation_index, rels.label_vi)); RLAB_EN = dict(zip(rels.relation_index, rels.label_en))
PID = dict(zip(rels.relation_index, rels.pid)); RID = {p: i for i, p in PID.items()}
print(f"08a: {N} thực thể, {R} quan hệ, {len(trip)} triple, {len(imgs)} ảnh; ngày chụp {collect['snapshot_date']}")

# Nhãn tiếng Việt cho vài khái niệm Wikidata chỉ có nhãn tiếng Anh (nhóm dịch; data card ghi rõ) và tách hai khái niệm cùng nhãn "tháp"
VI_LABEL_FIX = {"Q56242215": "nhà thờ chính tòa Công giáo", "Q186363": "kiến trúc Tân Gothic", "Q19683138": "khu Ramsar",
                "Q744373": "kiến trúc Tân Roman", "Q1459900": "danh sách dự kiến di sản thế giới",
                "Q43113623": "một phần của di sản thế giới UNESCO", "Q2232001": "hang động du lịch",
                "Q199451": "tháp (bảo tháp)", "Q12518": "tháp (công trình cao)"}
fix = ents.qid.isin(VI_LABEL_FIX)
ents.loc[fix, "label_vi"] = ents.loc[fix, "qid"].map(VI_LABEL_FIX)
ents.loc[fix & ents.label_vi_source.eq("en_fallback"), "label_vi_source"] = "team_translation"
ents.loc[fix & ents.label_vi_source.eq("wikidata_vi"), "label_vi_source"] = "wikidata_vi+disambiguated"
LAB = dict(zip(ents.entity_index, ents.label_vi))
dup = ents[ents.label_vi.duplicated(keep=False)]
if len(dup):
    print("Lưu ý: còn nhãn tiếng Việt trùng nhau:", dup[["qid", "label_vi"]].values.tolist())
