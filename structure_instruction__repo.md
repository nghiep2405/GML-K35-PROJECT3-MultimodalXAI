# Cấu trúc repo và hướng dẫn chạy Multimodal XAI

Source nằm trong `src/`. Dữ liệu đã xử lý và checkpoint của nhóm nằm trong `our_run_result/`. Lệnh `local` và `drive` đọc trực tiếp bundle này, ghi kết quả mới vào `runs/`. Dùng Python 3.11 hoặc 3.12 cho môi trường chạy mô hình. Các encoder truy vấn có thể cần tải model theo revision đã ghi trong JSON nên lần chạy đầu cần mạng.

## 1. Chạy local trên Windows PowerShell

### Cài đặt một lần

Mở PowerShell và chạy khối dưới. Nếu clone repo vào vị trí khác, đổi dòng đầu. Lệnh dùng Python 3.12 qua Python Launcher. Nếu máy đã có Python 3.11, đổi `-3.12` thành `-3.11`.

```powershell
$ProjectRoot = 'C:\Users\ADMIN\OneDrive - VNU-HCMUS\BSMS\Học máy dữ liệu đồ thị\GML-K35-PROJECT3-MultimodalXAI'
Set-Location -LiteralPath $ProjectRoot
py -3.12 -m venv .venv
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
& $Python -m pip install -e "./src[test]"
& $Python -m multimodal_xai.cli local --help

function Run-Xai {
    & $Python -m multimodal_xai.cli local @args
}
```

Trong terminal mới, khai báo lại `$ProjectRoot`, `$Python`, chạy `Set-Location` và khai báo hàm `Run-Xai`. Môi trường `.venv` đã tạo có thể dùng lại. Lệnh `local` tìm bundle từ thư mục hiện tại hoặc từ vị trí package cài editable. Nếu cần chọn một bản repo cụ thể, thêm `--root "$ProjectRoot"` vào lệnh.

### Dữ liệu đã có trong repo

| Pipeline | Dữ liệu được đọc trực tiếp | Checkpoint nhóm |
| --- | --- | --- |
| Flickr30k | `our_run_result/data/Flickr30k/` | `our_run_result/results/Flickr30k/3SEED/` và `ABLATION/` |
| DB15K | `our_run_result/data/DB15K/` | `our_run_result/results/DB15K/3SEED/seed_2027/`, `seed_2028/`, `seed_2029/` |
| FB15K-237 | `our_run_result/data/FB15K/` | `our_run_result/results/FB15K/3SEED/seed_2027/`, `seed_2028/`, `seed_2029/` |
| ViHeritage | `our_run_result/data/MMKG/` | Kiểm snapshot bằng manifest, không cần checkpoint. |
| Visual Genome | Tải riêng gói khoảng 12 GB theo README trong `our_run_result/data/VisualGeome/`. | Lệnh evaluate dùng checkpoint do lần train mới tạo ra. |

Các đường dẫn `phase2_relative_path` trong manifest DB15K tính từ `our_run_result/data/DB15K/`, của FB15K tính từ `our_run_result/data/FB15K/`. Để nguyên các file đã tải trong bundle. JSON encoder nằm cạnh checkpoint seed 2027. Snapshot mã phụ thuộc FB15K nằm trong `our_run_result/data/FB15K/sources/AdaMF-MAT/`.

### Kiểm tra nhanh ViHeritage

```powershell
Run-Xai viheritage verify
Get-Content -LiteralPath '.\runs\ViHeritage\verification.json'
```

Lệnh kiểm 19 artifact theo split manifest và ghi kết quả tại `runs/ViHeritage/verification.json`.

### Flickr30k

```powershell
& $Python -m pip install -e "./src[flickr,xai,test]"
Run-Xai flickr30k replay
```

Replay dùng năm checkpoint nhóm và xuất `runs/Flickr30k/replay/B2_reproduction_check.csv`. Để train mới, chạy từng lệnh dưới theo cấu hình bạn muốn. GPU giúp rút ngắn thời gian train.

```powershell
Run-Xai flickr30k train --seed 2026
Run-Xai flickr30k train --seed 2027
Run-Xai flickr30k train --seed 2028
Run-Xai flickr30k train --variant no_edge
Run-Xai flickr30k train --variant random_edges
Run-Xai flickr30k evaluate
Run-Xai flickr30k compare
```

Evaluate cần đủ năm run mới. Compare so với `our_run_result/results/Flickr30k/final_test_summary.json` và ghi `runs/Flickr30k/evaluate/parity.csv`. Để chạy `Run-Xai flickr30k explain`, cần bổ sung ảnh Flickr30k gốc trong `our_run_result/data/Flickr30k/raw_images/` và annotations mà workflow sử dụng.

### DB15K và FB15K-237

Cài dependency rồi chọn lệnh. Các lệnh replay/explain/quantify dùng ba checkpoint gốc trong bundle.

```powershell
& $Python -m pip install -e "./src[adamf,xai,test]"
Run-Xai db15k replay
Run-Xai fb15k237 replay
```

Replay xuất bảng C1 vào `runs/Replay/DB15K/` hoặc `runs/Replay/FB15K/`. Để tính XAI Level 2 và Level 3, chạy explain trước quantify cho cùng dataset.

```powershell
Run-Xai db15k explain
Run-Xai db15k quantify
Run-Xai fb15k237 explain
Run-Xai fb15k237 quantify
```

Level 2 xuất vào `runs/XAI-Level2/<dataset>/`. Level 3 lấy K đã chọn trên validation từ Level 2 và xuất vào `runs/XAI-Level3/<dataset>/`. Feature và bảng đường dẫn ảnh đủ cho tính điểm. Ảnh nguồn hoặc archive ảnh mà bảng tham chiếu cần được cung cấp thêm để hiển thị hình thật.

Để huấn luyện mới, dùng môi trường CUDA. Snapshot AdaMF-MAT dùng các dependency của dự án gốc, vì vậy quá trình chạy trên Windows cần được kiểm chứng với môi trường cụ thể.

```powershell
Run-Xai db15k train --seed 2027
Run-Xai db15k train-evaluate --seed 2027
Run-Xai fb15k237 train --seed 2027
Run-Xai fb15k237 train-evaluate --seed 2027
```

`train` dừng sau bước huấn luyện. `train-evaluate` tiếp tục train/resume rồi đánh giá khi run hoàn tất. Nếu phiên dừng do giới hạn session, chạy lại cùng lệnh và cùng seed để resume. Có thể chọn seed 2028 hoặc 2029. DB15K hỗ trợ thêm `--variant no_edges`, `--variant random_graph`, `--variant shuffled_edges_zero_overlap_v4` ở seed 2027.

Các checkpoint mới, config và test output nằm dưới `runs/DB15K/train/` hoặc `runs/FB15K/train/`. Mỗi lần đánh giá KG tạo `test_locked_*/` trong run. Các lệnh replay/XAI hiện dùng checkpoint nhóm trong bundle; chúng không tự chọn model vừa train.

### Visual Genome tải riêng

[Tải gói preprocess Visual Genome](https://drive.google.com/drive/folders/1qg9C6RcVA467jkjOj6-0tSuF1vLmUXJj?usp=drive_link).

Sau khi tải, đặt cây dữ liệu của gói tại `<repo>/dataset/visual_genome/` và `<repo>/experiment_v2/data/`. Đây là hai cây đầu vào của Visual Genome. Các bộ DB15K/FB15K vẫn được đọc trực tiếp trong bundle. Cần giữ toàn bộ shard, query pool và các manifest của gói.

```powershell
& $Python -m pip install -e "./src[visual-genome]"
Run-Xai visual-genome train --seed 2026
Run-Xai visual-genome evaluate --seed 2026
```

Seed chính là 2026/2027/2028. Có thể train `--variant no_edge` hoặc `--variant random_edges` ở seed 2026. Evaluator held-out hiện hỗ trợ main/no_edge. Các file mới nằm trong `runs/VisualGenome/experiment_v2/`. Chỉ đánh giá held-out sau khi checkpoint và alpha đã khóa.

## 2. Chạy trên Colab và Google Drive

Upload `src/` và `our_run_result/` nguyên cấu trúc vào **My Drive/MultimodalXAI/**. Gói Visual Genome được tải riêng nếu bạn cần pipeline đó.

```text
My Drive/MultimodalXAI/
  src/
    pyproject.toml
    multimodal_xai/
    notebooks/
  our_run_result/
    data/
      DB15K/
      FB15K/
      Flickr30k/
      MMKG/
      VisualGeome/README.md
    results/
      DB15K/
      FB15K/
      Flickr30k/
  dataset/visual_genome/              # từ gói VG tải riêng
  experiment_v2/data/                 # từ gói VG tải riêng
  runs/                              # chương trình tạo khi ghi kết quả
```

Chọn GPU runtime cho huấn luyện. Chạy cell mount và cài package theo component.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
# Ví dụ cho replay/XAI KG
%pip install -e "/content/drive/MyDrive/MultimodalXAI/src[adamf,xai]"
```

Chọn extra `[flickr,xai]` cho Flickr30k hoặc `[visual-genome]` cho VG. ViHeritage verify có thể dùng package cơ bản. Chọn và chạy các lệnh cần thiết.

```python
!multimodal-xai drive viheritage verify

!multimodal-xai drive db15k replay
!multimodal-xai drive fb15k237 replay
!multimodal-xai drive db15k explain
!multimodal-xai drive db15k quantify

!multimodal-xai drive db15k train-evaluate --seed 2027
!multimodal-xai drive fb15k237 train-evaluate --seed 2027
```

```python
!multimodal-xai drive flickr30k replay
!multimodal-xai drive flickr30k train --seed 2026
!multimodal-xai drive flickr30k train --seed 2027
!multimodal-xai drive flickr30k train --seed 2028
!multimodal-xai drive flickr30k train --variant no_edge
!multimodal-xai drive flickr30k train --variant random_edges
!multimodal-xai drive flickr30k evaluate
!multimodal-xai drive flickr30k compare
```

```python
!multimodal-xai drive visual-genome train --seed 2026
!multimodal-xai drive visual-genome evaluate --seed 2026
```

Lệnh drive dùng root mặc định `/content/drive/MyDrive/MultimodalXAI`. Khi đặt ở thư mục khác, thêm `--root "/đường/dẫn/MultimodalXAI"`. Đầu ra tự ghi vào `runs/` dưới root đó, theo đúng bảng bên dưới. Trên Molab có thể dùng các lệnh tương tự sau khi cài package và mount/tải bundle vào root tương ứng.

## 3. Các thành phần trong repo

| Đường dẫn | Tác dụng |
| --- | --- |
| `README.md`, `Project03_Multimodal_XAI.pdf` | Tổng quan và đề bài. |
| `doc/`, `report/`, `slide/`, `demo/` | Ghi chú, báo cáo LaTeX, slide và kịch bản demo. |
| `our_run_result/data/`, `our_run_result/results/` | Đầu vào đã xử lý và kết quả nhóm để đối chiếu. |
| `src/pyproject.toml` | Cài package, chọn dependency extras và tạo CLI. |
| `src/configs/` | Bản tham chiếu thiết lập Flickr30k và split Visual Genome. |
| `src/notebooks/` | Notebook wrapper theo pipeline, gọi các stage trong package. CLI local/drive tự cấu hình đường dẫn. Các wrapper chạy trực tiếp vẫn cần cấu hình tương ứng. |
| `src/multimodal_xai/cli.py`, `drive.py` | Parser và điều phối local/drive. `local` tự tìm repo, `drive` dùng đường dẫn Drive mặc định. |
| `src/multimodal_xai/artifacts/` | Ghi atomic, kiểm đường dẫn, tìm checkpoint và so metric Flickr30k. |
| `src/multimodal_xai/flickr30k/` | `data.py` đọc shard, `graph.py` dựng đồ thị, `model.py` định nghĩa SpatialGAT, `train.py` train/resume và `evaluate.py` đánh giá. |
| `src/multimodal_xai/retrieval/ranking.py` | Recall@K, MRR và quy tắc tie Flickr30k. |
| `src/multimodal_xai/visual_genome/` | `guard.py` bảo vệ split, `audit.py` kiểm dữ liệu, `pool.py` tạo query, `features.py` cache feature, `negatives.py` chọn negative, `sampler.py` lấy mẫu, `model.py` định nghĩa GAT, `train.py` train và `visualize.py` vẽ mẫu đồ thị. |
| `src/multimodal_xai/viheritage/snapshot.py` | Kiểm hash snapshot theo manifest. |
| `src/multimodal_xai/workflows/catalog.json`, `runner.py` | Ánh xạ notebook và cell sang stage thực thi. |
| `workflows/stages/shared_db15k/`, `shared_fb15k237/` | Adapter, cache, model, validation, train/resume và đánh giá của KG. |
| `workflows/stages/shared_kg_xai/` | `artifacts.py` đọc metadata, `models.py` scorer checkpoint, `ranking.py` xếp hạng, `explain.py` giải thích. |
| `workflows/stages/shared_visual_genome/` | Adapter dữ liệu và đồng bộ Molab. |
| `workflows/stages/<workflow>/` | Phần đặc thù từng notebook, tệp đánh số theo cell gốc. |
| `pytest.ini`, `test/`, `test-result/` | Cấu hình kiểm thử, các suite và log/manifest kiểm chứng. |
| `__init__.py` | Khai báo package Python. |

## 4. Đầu ra và đối chiếu với kết quả nhóm

| Lệnh | Đầu ra dưới `runs/` | Mốc nhóm |
| --- | --- | --- |
| Flickr30k replay | `Flickr30k/replay/B2_reproduction_check.csv` | `our_run_result/results/Flickr30k/final_test_summary.json` |
| Flickr30k train | `Flickr30k/train/<run>/` | Checkpoint và history nhóm theo seed/ablation. |
| Flickr30k evaluate/compare | `Flickr30k/evaluate/final_test_summary.json`, `parity.csv` | Final test summary của nhóm. |
| KG replay | `Replay/<dataset>/C1_reproduction.csv` và rank theo seed | `our_run_result/results/<dataset>/XAI/C1_test_ranks_seed*.csv` |
| KG explain/quantify | `XAI-Level2/<dataset>/`, `XAI-Level3/<dataset>/` | `our_run_result/results/<dataset>/XAI/`, `XAI_L3/` |
| KG train-evaluate | `DB15K/train/` hoặc `FB15K/train/`, gồm `test_locked_*/` | Theo checkpoint, seed và protocol tương ứng. |
| Visual Genome | `VisualGenome/experiment_v2/model/`, `reports/` | Dữ liệu/checkpoint VG tải riêng. |
| ViHeritage verify | `ViHeritage/verification.json` | `our_run_result/data/MMKG/processed/split_manifest.json` |

Nếu bundle không có `test_locked_*/`, C1 vẫn xuất rank replay và có thể ghi NaN ở cột đối chiếu locked gốc. Dùng bảng `C1_test_ranks_seed*.csv` đã lưu trong kết quả nhóm để đối chiếu theo query ID. Kết quả đạt kiểm tra đường dẫn/hash chưa chứng minh độ khớp metric. Chạy replay trên môi trường ML để kiểm điểm/rank thực tế.

## 5. Kiểm thử

```powershell
$run = "test-result/runs/local-cpu-$(Get-Date -Format yyyyMMdd-HHmmss)"
New-Item -ItemType Directory -Force -Path $run | Out-Null
& $Python -m pytest test --junitxml "$run/junit.xml"
# Có thể dùng unittest khi chưa cài pytest
& $Python -m unittest discover -s test -t . -v
```

CPU suite kiểm CLI, vị trí bundle, manifest và sampler. Test model cần Torch/PyG. Full train/replay cần các dependency của pipeline, GPU nếu workflow yêu cầu và bộ artifact đúng phiên bản.
