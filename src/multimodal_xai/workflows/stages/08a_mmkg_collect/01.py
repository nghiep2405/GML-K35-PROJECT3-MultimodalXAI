# A0. Cấu hình
import os, sys, re, json, glob, time, math, random, hashlib, shutil, tarfile, unicodedata, html, io, base64, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
for pkg, mod in [("sentence-transformers", "sentence_transformers"), ("requests", "requests")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np, pandas as pd, requests
from PIL import Image
from IPython.display import display, HTML

def find_project_root():
    env = os.environ.get("MMKG_PROJECT_ROOT")
    if not env and os.environ.get("MMKG_DATA_ROOT"):
        return Path.cwd()
    if env:
        return Path(env)
    pats = ["/content/drive/MyDrive/MLGs - Project", "/content/drive/MyDrive/*/MLGs - Project",
            "/content/drive/Shareddrives/*/MLGs - Project", "/content/drive/MyDrive/.shortcut-targets-by-id/*/MLGs - Project"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError("Không thấy thư mục 'MLGs - Project'. Thêm shortcut vào My Drive hoặc đặt MMKG_PROJECT_ROOT.")

PROJECT_ROOT = find_project_root()
MMKG = Path(os.environ.get("MMKG_DATA_ROOT", PROJECT_ROOT / "0. final" / "data" / "MMKG"))
RAW, PROC, IMG, REVIEW = MMKG / "raw", MMKG / "processed", MMKG / "images", MMKG / "review"
for d in [RAW, PROC, IMG, REVIEW]:
    d.mkdir(parents=True, exist_ok=True)
LOCAL = Path(os.environ.get("MMKG_LOCAL", "/content/mmkg_work" if IN_COLAB else "/tmp/mmkg_work"))
CACHE, CAND = LOCAL / "http_cache", LOCAL / "candidates"
for d in [CACHE, CAND]:
    d.mkdir(parents=True, exist_ok=True)

REFRESH = os.environ.get("MMKG_REFRESH", "0") == "1"   # True: bỏ cache, gọi lại API (dữ liệu có thể khác lần trước)
CONTACT = os.environ.get("MMKG_CONTACT", "")           # Wikimedia đề nghị User-Agent có cách liên hệ (email hoặc URL repo); điền nếu bị chặn 403
SEED = 2026                                            # seed chọn mẫu kiểm tra ảnh (khác seed huấn luyện 2027–2029)
TZ = timezone(timedelta(hours=7))

# Quy mô (mục 3.1 kế hoạch; đề gợi ý 50–150 thực thể, 300–1.000 triple, 200–500 ảnh)
COVERAGE_MIN = 80            # A1: ngưỡng đi tiếp
N_CANDIDATES = 150           # A2: số ứng viên đem đi lấy ảnh (dư để bù ứng viên không đủ ảnh)
N_IMG_ENT_MIN, N_IMG_ENT_MAX = 80, 88   # chừa chỗ cho thực thể không ảnh (tổng ≤ 150)
KIND_QUOTA = {"bridge": 6, "museum": 8}  # tối đa / loại, để KG không lệch về cầu hiện đại (ít quan hệ trên Wikidata)
USE_NEARBY, NEARBY_K, NEARBY_KM = True, 2, 5.0   # quan hệ suy ra từ tọa độ P625: 2 thực thể có ảnh gần nhất trong 5 km
USE_CENTURY = True                       # quan hệ suy ra từ năm xây P571: "xây dựng vào thế kỷ N"
N_NATURAL_MAX = 15           # tối đa số danh thắng thiên nhiên trong nhóm có ảnh
MAX_PER_ENTITY = 6           # ảnh cuối cùng / thực thể
MIN_PER_ENTITY = 2
TOTAL_IMG_MAX = 450
MAX_FILES_INFO = 150         # số file Commons tối đa đọc metadata / thực thể
MAX_DL_PER_ENTITY = 10       # số ảnh tải về / thực thể trước lọc CLIP
DEPTH1_IF_BELOW = 4          # category độ sâu 0 có < 4 ảnh đạt thì xuống độ sâu 1
MAX_SUBCATS = 15
MIN_SHORT_SIDE = 400
MAX_ASPECT = 2.5             # bỏ ảnh toàn cảnh quá dài (CLIP cắt giữa sẽ mất nội dung)
THUMB_WIDTH = 960            # kích thước thumbnail chuẩn của Wikimedia (không dùng 640: từ 2026 kích thước lạ bị giới hạn)
NEG_PROB_MAX = 0.5           # A4: bỏ ảnh nếu tổng xác suất các nhãn "không phải ảnh công trình/cảnh quan" ≥ 0.5
DUP_COS = 0.95
MIN_TAIL_FREQ = 2            # thực thể không ảnh phải xuất hiện ≥ 2 lần
MAX_ENTITIES = 150
N_REVIEW = 100
API_SLEEP, DL_SLEEP = 0.25, 0.6
SNAPSHOT = datetime.now(TZ).strftime("%Y-%m-%d")          # ghi đè ở dưới nếu chạy lại từ cache

UA = "ViHeritageMMKG/1.0 (HCMUS Machine Learning with Graphs coursework, Project 03" + (f"; {CONTACT})" if CONTACT else ")")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
WD_API, COMMONS_API, VIWIKI_API = "https://www.wikidata.org/w/api.php", "https://commons.wikimedia.org/w/api.php", "https://vi.wikipedia.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"

# Cache HTTP: đọc lại bản đã lưu trên Drive (nếu có) để lần chạy sau tất định
CACHE_TAR = RAW / "http_cache.tar.gz"
CAND_TAR = RAW / "commons_candidate_images.tar"      # ảnh ứng viên đã tải (để chạy lại cho đúng từng byte; không phân phối)
if not REFRESH:
    for tar_p, dst in [(CACHE_TAR, CACHE), (CAND_TAR, CAND)]:
        if tar_p.exists() and not any(dst.iterdir()):
            with tarfile.open(tar_p) as t:
                t.extractall(dst)
SNAP_FILE = RAW / "snapshot_date.txt"
if not REFRESH and any(CACHE.iterdir()):      # chạy lại từ cache: ngày chụp = ngày của phản hồi cũ nhất trong cache
    SNAPSHOT = (SNAP_FILE.read_text().strip() if SNAP_FILE.exists() else
                datetime.fromtimestamp(min(p.stat().st_mtime for p in CACHE.iterdir()), TZ).strftime("%Y-%m-%d"))
SNAP_FILE.write_text(SNAPSHOT)
print("MMKG:", MMKG, "| cache:", sum(1 for _ in CACHE.iterdir()), "phản hồi | snapshot:", SNAPSHOT)
