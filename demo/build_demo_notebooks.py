"""Dựng 3 notebook web demo (một cấp độ một notebook), cùng một giao diện multimodal_xai/demo_web.py.

  demo_cap1_visual_genome.ipynb  : CLIP + GAT trên scene graph Visual Genome (lấy các ô nạp dữ liệu, mô hình của notebook 11)
  demo_cap2_db15k_fb15k.ipynb    : CLIP + AdaMF trên DB15K, FB15K-237-IMG (chạy các ô C0, C0b, C0c, C10 của notebook 07)
  demo_cap3_viheritage.ipynb     : CLIP + GAT / AdaMF trên ViHeritage-MMKG (ô X0 của 10b, 10c, 10d)

Chạy: python src/notebooks/build_vg_xai_notebook.py  (nếu vừa sửa notebook 11), rồi python demo/build_demo_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent          # demo/
SRC = HERE.parent / "src"
NB = SRC / "notebooks"
DEMO_WEB = (SRC / "multimodal_xai" / "demo_web.py").read_text(encoding="utf-8")
EXPORTS = ["PALETTE", "PAGE_HEIGHT", "image_data_url", "rank_text", "removal_sentence", "node", "link",
           "results_page", "results_iframe", "message_html", "launch"]
DEMO_WEB_CELL = (DEMO_WEB.rstrip() + "\n\n# Gom các hàm của giao diện vào W (giống `from multimodal_xai import demo_web as W` ở notebook cấp 2)\n"
                 "import types as _types\nW = _types.SimpleNamespace(" + ", ".join(f"{n}={n}" for n in EXPORTS) + ")\n")
RUN_NOTE = """
**Cách dùng khi quay video:** Runtime → Run all. Ô cuối in ra một link `https://….gradio.live`: mở link ở tab mới (toàn màn hình).
Chọn ca trong danh sách là trang tự chạy; bấm **Bỏ giải thích S** trên đồ thị để xem ảnh rơi hạng. Link công khai và tự hết hạn,
tắt notebook (Runtime → Disconnect) sau khi quay. Đặt biến môi trường `XAI_DEMO_SHARE=0` nếu chỉ muốn xem trong notebook.
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def write(name, cells, gpu=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
    nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Đã ghi", HERE / name)


# ---------------- cấp 1: Visual Genome ----------------
nb11 = json.loads((NB / "11_visualgenome_gat_xai_level1.ipynb").read_text(encoding="utf-8"))
SRC11 = ["".join(c["source"]) for c in nb11["cells"] if c["cell_type"] == "code"]


def cell11(marker):
    hits = [s for s in SRC11 if marker in s]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


fetch = cell11("def fetch_image(image_id)")
fetch = fetch[:fetch.index("def explain_image(")].rstrip()
install = cell11("'pip', 'install'").replace("'requests'])", "'requests', 'gradio'])")
write("demo_cap1_visual_genome.ipynb", [
    md("# Demo cấp 1 · CLIP + GAT trên scene graph Visual Genome\n\n"
       "Nhập câu (hoặc chọn một trong 10 ca của báo cáo) → top-k ảnh sau khi xếp hạng lại → đồ thị giải thích S của từng ảnh "
       "(các quan hệ của scene graph làm tăng điểm đồ thị nhiều nhất, K* đọc từ kết quả notebook 11) → kiểm chứng: bỏ S hoặc bỏ cùng "
       "số quan hệ ngẫu nhiên rồi xếp hạng lại. Mô hình là checkpoint chính đã khóa; không huấn luyện, không chọn gì trên test.\n"
       + RUN_NOTE),
    code(install),
    md("## Cấu hình, dữ liệu, mô hình (giống notebook 11)"),
    code(cell11("def find_final_root")),
    code(cell11("compact = torch.load")),
    code(cell11("class EdgeAwareGAT")),
    code((NB / "vg_xai_core.py").read_text(encoding="utf-8")),
    code(cell11("global_to_local = torch.full")),
    code((NB / "vg_xai_plot.py").read_text(encoding="utf-8")),
    code(fetch),
    md("## Giao diện dùng chung (nhúng nguyên văn `src/multimodal_xai/demo_web.py`)"),
    code(DEMO_WEB_CELL),
    md("## Web demo"),
    code((HERE / "web_cap1_visual_genome.py").read_text(encoding="utf-8")),
], gpu=True)

# ---------------- cấp 2: DB15K, FB15K-237-IMG ----------------
SETUP2 = r"""
# Nạp package multimodal_xai: ưu tiên bản trên Drive (thư mục src có demo_web.py), nếu chưa có thì lấy từ GitHub.
import os, sys, glob, subprocess
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")

def find_src():
    cands = [os.environ.get("MXAI_SRC", ""), "/content/drive/MyDrive/MultimodalXAI/src"]
    cands += sorted(glob.glob("/content/drive/MyDrive/*GML-K35-PROJECT3-MultimodalXAI*/src"))
    cands += sorted(glob.glob("/content/drive/MyDrive/*/*GML-K35-PROJECT3-MultimodalXAI*/src"))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "multimodal_xai", "demo_web.py")):
            return c
    # Không có trên Drive: tải mã từ GitHub, bỏ qua file dữ liệu Git LFS; mã demo nằm ở nhánh `demo` nếu chưa gộp vào main.
    dst, env = "/content/GML-K35-PROJECT3-MultimodalXAI", dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not os.path.exists(dst):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/nghiep2405/GML-K35-PROJECT3-MultimodalXAI.git", dst],
                       check=True, env=env)
    if not os.path.exists(os.path.join(dst, "src", "multimodal_xai", "demo_web.py")):
        subprocess.run(["git", "-C", dst, "fetch", "-q", "--depth", "1", "origin", os.environ.get("MXAI_BRANCH", "demo")], check=True, env=env)
        subprocess.run(["git", "-C", dst, "checkout", "-q", "FETCH_HEAD"], check=True, env=env)
    return os.path.join(dst, "src")

SRC = find_src(); sys.path.insert(0, SRC)
for pkg, mod in [("transformers", "transformers"), ("pyarrow", "pyarrow"), ("gradio", "gradio")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import multimodal_xai.demo_web
print("multimodal_xai:", SRC)
"""
run = lambda i: code(f"from multimodal_xai.workflows.runner import run_stage\nrun_stage('07_db15k_fb15k_adamf_xai_level2', {i}, globals())")
write("demo_cap2_db15k_fb15k.ipynb", [
    md("# Demo cấp 2 · CLIP + AdaMF trên DB15K và FB15K-237-IMG\n\n"
       "Chạy đúng các ô C0, C0b, C0c (cấu hình, dữ liệu, checkpoint) và C10 (demo) của notebook "
       "`kg_xai/07_db15k_fb15k_adamf_xai_level2.ipynb`; K* và 10 ca đọc từ kết quả C6, C7 đã lưu trên Drive. "
       "Truy vấn gián tiếp: thực thể neo + quan hệ; đồ thị là các bộ ba train làm tăng điểm của thực thể trả về nhiều nhất "
       "(học lại nhúng kiểu Kelpie), kiểm chứng bằng bỏ S hoặc bỏ cùng số bộ ba ngẫu nhiên rồi xếp hạng lại.\n" + RUN_NOTE),
    code(SETUP2), run(1), run(2), run(3),
    md("## Web demo"), run(22), run(23),
], gpu=True)

# ---------------- cấp 3: ViHeritage-MMKG ----------------
write("demo_cap3_viheritage.ipynb", [
    md("# Demo cấp 3 · ViHeritage-MMKG (tiếng Việt, nhóm tự xây)\n\n"
       "Dùng đúng mã dùng chung X0 của notebook 10b, 10c, 10d (dữ liệu 08b, checkpoint 09 đã khóa, hai bộ giải thích). "
       "Chọn mô hình (GAT: che cạnh; AdaMF: học lại nhúng), cách diễn đạt câu (câu hỏi, câu mô tả, tiếng Anh) và một ca của báo cáo, "
       "hoặc tự nhập câu + thực thể neo + quan hệ. Đồ thị là các bộ ba train làm tăng điểm nhiều nhất (K* đọc từ `10b_run_info.json`).\n"
       + RUN_NOTE),
    code("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'], check=True)"),
    md("## X0. Mã dùng chung của 10b, 10c, 10d (nguyên văn)"),
    code((NB / "viheritage" / "mmkg_xai_shared_x0.py").read_text(encoding="utf-8")),
    md("## Giao diện dùng chung (nhúng nguyên văn `src/multimodal_xai/demo_web.py`)"),
    code(DEMO_WEB_CELL),
    md("## Web demo"),
    code((HERE / "web_cap3_viheritage.py").read_text(encoding="utf-8")),
])
