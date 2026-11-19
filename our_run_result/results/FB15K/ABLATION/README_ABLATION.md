# Ablation cấp 2 — FB15K-237 (AdaMF bản truy hồi), seed 2027

Đề, mục 7, Nhiệm vụ 2: bỏ cạnh, xáo cạnh hoặc thay bằng đồ thị ngẫu nhiên để kiểm tra phần cải thiện có đến từ cấu trúc quan hệ hay không.
Mỗi biến thể huấn luyện lại với cùng cấu hình như mô hình chính, chọn epoch và α trên validation, rồi chạy test một lần với α đã khóa
(xem `test_protocol.json` trong từng thư mục).

| Biến thể | α (val) | MRR gộp | MRR trực tiếp | MRR gián tiếp | Giữ lại mức tăng trực tiếp | Giữ lại mức tăng gián tiếp |
|---|---|---|---|---|---|---|
| Mô hình đầy đủ (seed 2027) | 0,1 | 0,2896 | 0,7751 | 0,1198 | 100% | 100% |
| no_edges | 1,0 | 0,0991 | 0,3475 | 0,0122 | 0% | 0% |
| shuffled_edges | 0,1 | 0,2878 | 0,7894 | 0,1124 | 103,3% | 93,1% |
| random_graph | 0,1 | 0,2869 | 0,7867 | 0,1121 | 102,7% | 92,9% |
| CLIP thuần | — | 0,0991 | 0,3475 | 0,0122 | — | — |

"Giữ lại mức tăng" = (MRR của biến thể − CLIP) / (MRR của mô hình đầy đủ − CLIP), tính riêng nhóm trực tiếp và gián tiếp.

**Đọc kết quả.** Bỏ cạnh thì validation chọn α = 1 (về đúng CLIP). Nhưng xáo cạnh và đồ thị ngẫu nhiên vẫn giữ khoảng 93% mức tăng
gián tiếp và toàn bộ mức tăng trực tiếp. Bản FB15K học bằng loss truy hồi trên truy vấn huấn luyện; ablation đổi anchor/quan hệ của truy vấn
nhưng giữ ảnh đích, nên mô hình vẫn học được thực thể nào khớp ảnh nào. Vì vậy trên FB15K, phần lớn mức tăng **không** chứng minh được là
nhờ đúng cấu trúc quan hệ. Cần nói thẳng điều này trong báo cáo; DB15K là bằng chứng chính cho vai trò của cạnh thật.

**Giới hạn.** Mỗi biến thể mới có 1 seed (2027), trong khi đề yêu cầu ≥ 3 seed cho thực nghiệm chính. Ghi rõ trong báo cáo.

## Tệp
- `FB15K_ABLATION_summary_seed2027.csv`: bảng trên, kèm R@1, R@10 nhóm gián tiếp.
- `FB15K_ablation_test_metrics_all.csv`: toàn bộ test_metrics của 4 run (đầy đủ + 3 ablation), chuẩn hóa tên nhóm, có cột `source`.
- Thư mục con `<biến thể>_seed2027/`: `test_metrics.csv`, `test_protocol.json` (bản sao nguyên văn).
- Xếp hạng từng truy vấn (`test_query_ranks.csv`) và checkpoint vẫn ở thư mục gốc (cột `source`, gốc `MLGs - Project/`).

## Nguồn (dưới `MLGs - Project/dataset/adamf/model/adamf_v1/`)
- Đầy đủ: `fb15k237_clean_kgc_to_retrieval_50ep_seed2027_v1/seed_2027/test_locked_20261102T201722_386357Z`
- no_edges: `fb15k237_clean_kgc_to_retrieval_seed2027_ablation_no_edges_v1/seed_2027/results/test_locked_20261104T083311_581338Z`
- shuffled_edges: `fb15k237_clean_kgc_to_retrieval_seed2027_ablation_shuffled_edges_v1/seed_2027/results/test_locked_20261113T074309_287872Z`
- random_graph: `fb15k237_clean_kgc_to_retrieval_seed2027_ablation_random_graph_v1/seed_2027/results/test_locked_20261104T204145_134973Z`

Mỗi run lấy thư mục `test_locked_*` mới nhất. Run random_graph có 5 thư mục `test_locked_*` (chạy test nhiều lần trên cùng checkpoint);
nên kiểm tra lại các lần đó cho cùng số liệu, để khẳng định không chọn kết quả trên test.

Tổng hợp ngày 15/11/2026 từ các file trên; không chạy lại mô hình.
