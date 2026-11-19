# Ablation cấp 2 — DB15K (AdaMF bản KGC), seed 2027

Đề, mục 7, Nhiệm vụ 2: bỏ cạnh, xáo cạnh hoặc thay bằng đồ thị ngẫu nhiên để kiểm tra phần cải thiện có đến từ cấu trúc quan hệ hay không.
Mỗi biến thể huấn luyện lại với cùng cấu hình như mô hình chính, chọn epoch và α trên validation, rồi chạy test một lần với α đã khóa
(xem `test_protocol.json` trong từng thư mục).

| Biến thể | α (val) | MRR gộp | MRR trực tiếp | MRR gián tiếp | Giữ lại mức tăng gián tiếp |
|---|---|---|---|---|---|
| Mô hình đầy đủ (seed 2027) | 0,0 | 0,4145 | 0,5520 | 0,3164 | 100% |
| no_edges | 1,0 | 0,2577 | 0,5520 | 0,0475 | 0% |
| shuffled_edges (zero_overlap_v4) | 0,2 | 0,3156 | 0,5520 | 0,1468 | 36,9% |
| random_graph | 1,0 | 0,2577 | 0,5520 | 0,0475 | 0% |
| CLIP thuần | — | 0,2577 | 0,5520 | 0,0475 | — |

"Giữ lại mức tăng gián tiếp" = (MRR gián tiếp của biến thể − CLIP) / (MRR gián tiếp của mô hình đầy đủ − CLIP).
Truy vấn trực tiếp của DB15K dùng CLIP (direct_policy = CLIP_only), nên không đổi giữa các biến thể.

**Đọc kết quả.** Bỏ cạnh hoặc dùng đồ thị ngẫu nhiên thì validation chọn α = 1, tức đồ thị không còn giúp gì. Xáo cạnh (giữ tần suất
quan hệ và bậc) chỉ giữ 37% mức tăng ở nhóm gián tiếp. Phần lớn cải thiện cần đúng cạnh thật.

**Giới hạn.** Mỗi biến thể mới có 1 seed (2027), trong khi đề yêu cầu ≥ 3 seed cho thực nghiệm chính. Ghi rõ trong báo cáo.

## Tệp
- `DB15K_ABLATION_summary_seed2027.csv`: bảng trên, kèm R@1, R@10 nhóm gián tiếp.
- `DB15K_ablation_test_metrics_all.csv`: toàn bộ test_metrics của 4 run (đầy đủ + 3 ablation), chuẩn hóa tên nhóm, có cột `source`.
- Thư mục con `<biến thể>_seed2027/`: `test_metrics.csv`, `test_protocol.json` (bản sao nguyên văn).
- Xếp hạng từng truy vấn (`test_query_ranks.csv`) và checkpoint vẫn ở thư mục gốc (cột `source`, gốc `MLGs - Project/`).

## Nguồn (dưới `MLGs - Project/dataset/adamf/model/adamf_v1/`)
- Đầy đủ: `db15k_clean_kgc_to_retrieval_50ep_seed2027_v1/test_locked_20261102T150657_303904Z`
- no_edges: `db15k_clean_kgc_to_retrieval_seed2027_ablation_no_edges_v1/seed_2027/results/test_locked_20261104T205920_003731Z`
- shuffled_edges: `db15k_clean_kgc_to_retrieval_seed2027_ablation_shuffled_edges_zero_overlap_v4/seed_2027/results/test_locked_20261115T082857_557259Z`
- random_graph: `db15k_clean_kgc_to_retrieval_seed2027_ablation_random_graph_v1/seed_2027/results/test_locked_20261104T221026_435049Z`

Bản xáo cạnh chính thức là v4 (zero_overlap: chặn mọi trùng với triple val/test). Các bản v1–v3 (`shuffled_edges_v1`, `_split_guard_v2`,
`_train_only_v3`) là các lần thử trước, không dùng. Mỗi run lấy thư mục `test_locked_*` mới nhất.

Tổng hợp ngày 15/11/2026 từ các file trên; không chạy lại mô hình.
