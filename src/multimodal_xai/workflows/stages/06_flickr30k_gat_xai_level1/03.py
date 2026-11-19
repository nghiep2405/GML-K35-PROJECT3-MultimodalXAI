# ===================== CẤU HÌNH (chỉ sửa ở đây nếu cần) =====================
FINAL_ROOT = os.environ.get("XAI_FINAL_ROOT")      # vd: "/content/drive/MyDrive/MLGs - Project/0. final"
IMAGE_DIR = os.environ.get("XAI_IMAGE_DIR")        # thư mục ảnh Flickr30k *.jpg; để trống thì notebook tự tải (xem B7b)
DOWNLOAD_TEST_IMAGES = os.environ.get("XAI_DOWNLOAD_TEST_IMAGES", str(int(IN_COLAB))) == "1"                    # tự tải 1000 ảnh test từ Hugging Face (nlphuji/flickr30k) nếu chưa có ảnh
WRITE_TO_DRIVE = os.environ.get("XAI_WRITE_DRIVE", "0") == "1"                              # True: ghi kết quả vào 0. final/results/Flickr30k/XAI
MAIN_RUN = "seed_2026"                             # mô hình dùng để giải thích chính
N_VAL_QUERIES_B6 = int(os.environ.get("XAI_N_VAL", "300"))   # số truy vấn validation để chọn K
K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
COMPREHENSIVENESS_TARGET = 0.80                    # K nhỏ nhất mà xóa K quan hệ đầu gây >= 80% mức giảm do xóa mọi quan hệ có ích (trung vị, val)
N_RANDOM_REPEATS_B10 = 20                          # số lần bỏ ngẫu nhiên cho mỗi truy vấn ở B10
SELECTION_SEED = 2026                              # seed cố định cho mọi lựa chọn ngẫu nhiên
# ===========================================================================

from pathlib import Path
import glob

def _find_final_root():
    if FINAL_ROOT:
        return Path(FINAL_ROOT).expanduser().resolve()
    patterns = [
        "/content/drive/MyDrive/0. final",
        "/content/drive/MyDrive/*/0. final",
        "/content/drive/MyDrive/*/*/0. final",
        "/content/drive/MyDrive/*/*/*/0. final",
        "/content/drive/Shareddrives/*/0. final",
        "/content/drive/Shareddrives/*/*/0. final",
    ]
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if (Path(candidate) / "data/Flickr30k/processed/clip_vit_b32_openai").exists():
                return Path(candidate)
    raise FileNotFoundError(
        "Không tìm thấy '0. final'. Nếu thư mục được chia sẻ cho bạn, hãy 'Add shortcut to Drive' "
        "rồi điền FINAL_ROOT ở ô cấu hình."
    )

FINAL = _find_final_root()
CKPT_DIRS = {
    "seed_2026": ("3SEED/seed_2026", "real_edges", 2026),
    "seed_2027": ("3SEED/seed_2027", "real_edges", 2027),
    "seed_2028": ("3SEED/seed_2028", "real_edges", 2028),
    "no_edge": ("ABLATION/no_edge", "no_edge", 2026),
    "random_edges": ("ABLATION/random_edges", "random_edges", 2026),
}
for name, (sub, _, _) in CKPT_DIRS.items():
    assert (FINAL / "results/Flickr30k" / sub / "best_model.pt").exists(), f"Thiếu checkpoint {name}"
assert (FINAL / "data/Flickr30k/processed/clip_vit_b32_openai/images/test").exists(), "Thiếu dữ liệu test"

WORK = Path("/content/xai_work") if IN_COLAB else Path(os.environ.get("XAI_WORK", str(Path.cwd() / "test-result/runs/flickr-xai-work")))
OUT = (FINAL / "results/Flickr30k/XAI") if (WRITE_TO_DRIVE and IN_COLAB) else Path(
    os.environ.get("XAI_OUT", str(WORK / "XAI")))
for folder in (WORK, OUT, OUT / "figures", OUT / "cases"):
    folder.mkdir(parents=True, exist_ok=True)
print("FINAL :", FINAL)
print("OUT   :", OUT)
