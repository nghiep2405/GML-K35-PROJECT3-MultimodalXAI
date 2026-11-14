# BẢNG MA TRẬN MINH CHỨNG VÀ KHOẢNG TRỐNG THỰC NGHIỆM (EVIDENCE GAP MAP)
## DỰ ÁN: MULTIMODAL RETRIEVAL WITH GRAPH-BASED EXPLAINABILITY (PROJECT 03)

> **Mục đích tài liệu:** Đây là tài liệu tác nghiệp nội bộ dành cho người viết và nhóm nghiên cứu, theo dõi đối chiếu từng mục trong sườn báo cáo 7 chương với artifact thực tế, trạng thái kiểm chứng và các việc cần bổ sung. **Không đưa nguyên bảng tracker này vào thân báo cáo chính.**
> 
> **Quy ước trạng thái (Evidence Status):**
> - `VERIFIED`: Đã có artifact thực tế (file/code/metrics/checkpoint/audit), đã được kiểm chứng độc lập hoặc có số liệu tính toán nhất quán.
> - `PARTIAL`: Đã có code/notebook/kết quả sơ bộ nhưng chưa đủ số seed, chưa có đầy đủ phân tích định lượng hoặc đang trong quá trình rà soát.
> - `NOT VERIFIED`: Mới có định hướng trong đề tài, thư mục trống hoặc chưa có artifact kiểm chứng độc lập.
> - `SOURCE NOT ACCESSIBLE`: Tài liệu tham chiếu nằm trên Google Drive hoặc môi trường ngoài không có quyền truy cập trực tiếp từ local.

---

### Bảng theo dõi chi tiết theo từng chương mục

| Chương / Mục | Nội dung / Tuyên bố dự kiến (Intended Claim) | Artifact / File nguồn | Evidence Status | Nội dung cần bổ sung / Hoàn thiện (What to Add) | Mức ưu tiên |
|---|---|---|:---:|---|:---:|
| **Abstract** | Tóm tắt toàn diện vấn đề, 3 cấp độ, kết quả verified và giới hạn học thuật | `report/Appendix/summary.tex` | `VERIFIED` | Cập nhật số liệu cuối cùng khi hoàn tất Cấp độ 3 | P1 |
| **1.1** | Bối cảnh truy hồi đa phương thức, vai trò CLIP và khoảng trống quan hệ | `Project03_Multimodal_XAI.pdf (Mục 1--2)` | `VERIFIED` | Trích dẫn bổ sung các bài báo 2024--2026 về giới hạn của Dual-Encoder | P2 |
| **1.2** | Phát biểu bài toán: thiếu quan hệ và tính giải thích trong cosine similarity | `Project03_Multimodal_XAI.pdf (Mục 1, Nhiệm vụ 1--3)` | `VERIFIED` | Minh họa ví dụ truy vấn gián tiếp *child - plays - football* | P2 |
| **1.3** | Ba câu hỏi nghiên cứu RQ1, RQ2, RQ3 và mối quan hệ phụ thuộc | `Project03_Multimodal_XAI.pdf (Mục 1, 5)` | `VERIFIED` | Hoàn thiện sơ đồ liên kết phụ thuộc `research_questions_flow` | P1 |
| **1.4** | Phạm vi text-to-image, các tập dữ liệu, đóng góp chính và giới hạn | `Project03_Multimodal_XAI.pdf (Mục 3, 5, 8)` | `VERIFIED` | Khóa thông tin thành viên nhóm trong Phụ lục D | P1 |
| **2.1** | Cơ sở truy vấn đa phương thức: CLIP, cosine, Recall@K, MRR | `radford2021clip; Project03_Multimodal_XAI.pdf` | `VERIFIED` | Bổ sung nDCG nếu triển khai phần mở rộng | P2 |
| **2.2** | Học biểu diễn đồ thị: Scene Graph, MMKG, GCN, GAT, GraphSAGE | `velickovic2018gat; kipf2017gcn; krishna2017` | `VERIFIED` | Công thức attention GAT và cơ chế truyền thông điệp | P2 |
| **2.3** | Cải thiện truy vấn bằng đồ thị: candidate pool top-$k$, score fusion $\alpha$ | `Project03_Multimodal_XAI.pdf (Mục 4.2, 5)` | `VERIFIED` | Trích dẫn các nghiên cứu reranking đa phương thức | P2 |
| **2.4** | MMKG nâng cao: AdaMF-MAT, NativE; phân biệt rõ bài báo gốc vs bản triển khai | `adamfmat2024; native2024sigir` | `VERIFIED` | Sơ đồ phân định phạm vi kiến trúc AdaMF gốc vs AdaMF-based của nhóm | P1 |
| **2.5** | Khả năng giải thích đồ thị: đường đi/đồ thị con; Fidelity, Validity, Sparsity | `ying2019gnnexplainer; yuan2021subgraphx` | `VERIFIED` | Định nghĩa toán học của $\text{Fidelity}^-$ và $\text{Sparsity}$ | P1 |
| **2.6** | Động lực nghiên cứu: khoảng trống dẫn dắt tiến trình L1 $\rightarrow$ L2 $\rightarrow$ L3 | `Project03_Multimodal_XAI.pdf (Mục 1--3)` | `VERIFIED` | Lập luận chặt chẽ tránh tự nhận định novelty khi chưa khảo sát đủ | P2 |
| **3.1** | Sơ đồ nghiên cứu tổng quát 3 tầng (CLIP $\rightarrow$ Graph Rerank $\rightarrow$ XAI) | `Project03_Multimodal_XAI.pdf; plan_Nghiep_Khanh.md` | `VERIFIED` | Vẽ hình sơ đồ kiến trúc tổng thể chính thức | **P0** |
| **3.2.1** | CLIP baseline: ViT-B/32 đóng băng, tính điểm cosine zero-shot | `notebook/03_3_clip_baseline_retrieval.ipynb` | `VERIFIED` | Khóa mã băm embedding trích xuất | P1 |
| **3.2.2** | Xây dựng đồ thị tĩnh: thực thể, nhãn đối tượng, quan hệ cục bộ | `notebook/02_2_graph_construction.ipynb; graph_manifest.json` | `VERIFIED` | Bổ sung bảng thống kê số đỉnh/cạnh trung bình mỗi ảnh | P1 |
| **3.2.3** | Graph Attention Network: EdgeAwareGAT module, readout pooling | `04a_gat_training_validation.ipynb` | `VERIFIED` | Mã nguồn kiến trúc EdgeAwareGAT | P1 |
| **3.2.4** | Dung hợp điểm CLIP--Đồ thị: định nghĩa $\alpha_{\text{CLIP}}$, chọn trên validation | `04a_gat_training_validation.ipynb; audit Flickr30k` | `VERIFIED` | Đồ thị quét siêu tham số $\alpha$ trên validation set | P1 |
| **3.2.5** | Giải thích đồ thị mức cơ bản: trích xuất attention weights lớp cuối | `notebook/06_flickr30k_gat_xai_level1.ipynb` | `PARTIAL` | Trích xuất hình ảnh minh họa attention cho ví dụ cụ thể | P1 |
| **Trans L1$\rightarrow$L2** | Giới hạn đồ thị tĩnh cục bộ, động lực mở rộng sang MMKG đa quan hệ | `Project03_Multimodal_XAI.pdf` | `VERIFIED` | Đoạn văn chuyển tiếp lập luận học thuật liên kết | P1 |
| **3.3.1** | Động lực học đồ thị nâng cao: đa quan hệ ngữ nghĩa, bù đắp đa phương thức | `adamfmat2024` | `VERIFIED` | Phân tích sự bất đối xứng giữa text và image trong KG | P2 |
| **3.3.2** | Biểu diễn MMKG: bộ ba $(h, r, t)$ gắn nhúng text và ảnh thực thể | `dataset/adamf/manifest.json; DB15K/FB15k manifests` | `VERIFIED` | Mô tả chi tiết schema lưu trữ Parquet và index mapping | P1 |
| **3.3.3** | Dung hợp đa phương thức dựa trên AdaMF: cơ chế gating thích nghi | `src/gml_v2/; 04b2_adamf_*.py` | `VERIFIED` | Công thức chi tiết của mạng cổng gating $\gamma_T, \gamma_V$ | P1 |
| **3.3.4** | Tăng cường truy vấn và xếp hạng lại: hybrid scoring, $\alpha_{\text{CLIP}}=0.2$ | `DB15K_Shuffled_Edges_v2_Review_Feedback.md` | `VERIFIED` | Khẳng định quy ước ký hiệu $\alpha_{\text{CLIP}}$ nhất quán | **P0** |
| **3.3.5** | Xử lý truy vấn trực tiếp vs. gián tiếp: phân định định nghĩa và bối cảnh | `test_protocol.json; Project03_Multimodal_XAI.pdf` | `PARTIAL` | Chuẩn hóa danh sách bộ 20 truy vấn đánh giá (10 direct, 10 indirect) | **P0** |
| **Trans L2$\rightarrow$L3** | Hiệu năng cao cần đi kèm giải thích trung thực với scoring và chuyển giao | `Project03_Multimodal_XAI.pdf` | `VERIFIED` | Đoạn văn chuyển tiếp lập luận học thuật liên kết | P1 |
| **3.4.1** | Động lực giải thích: vượt qua hộp đen, chống ngụy biện hậu nghiệm | `Project03_Multimodal_XAI.pdf` | `VERIFIED` | Yêu cầu giải thích bắt nguồn từ hàm chấm điểm | P2 |
| **3.4.2** | Sinh giải thích dựa trên đồ thị: trích xuất đường đi suy luận bộ ba | `notebook/07_db15k_fb15k_adamf_xai_level2.ipynb` | `PARTIAL` | Mã giả thuật toán trích xuất đồ thị con tối thiểu | P1 |
| **3.4.3** | Đánh giá Fidelity, Validity, Sparsity: giao thức làm nhiễu đồ thị | `Project03_Multimodal_XAI.pdf (Mục 5, 7)` | `VERIFIED` | Quy trình chuẩn hóa đo lường can thiệp xóa cạnh | P1 |
| **3.4.4** | Xây dựng Custom MMKG: miền hẹp tiếng Việt, 50--150 entities, 300--1k triples | `dataset/mmkg_tv` | `PARTIAL` | Thu thập hoàn chỉnh ảnh, metadata và tạo Data Card | P1 |
| **3.4.5** | Chuyển giao sang miền đặc thù: quy trình áp dụng mô hình tốt nhất | `0. final/results/MMKG` | `NOT VERIFIED` | Thiết lập script benchmark trên tập dữ liệu tự xây | **P0** |
| **4.1** | Đặc trưng các tập dữ liệu: Flickr30k, DB15K, FB15k-237, VG, Custom MMKG | `dataset/ manifests; plan_Nghiep_Khanh.md` | `VERIFIED` | Ghi chú rõ nguyên tắc không so sánh chéo metric giữa các tập | P1 |
| **4.2** | Tiền xử lý và chia tập: split_v2 phân cụm ảnh trùng, candidate pool top-1000 | `frozen_contract.json; test_protocol.json` | `VERIFIED` | Báo cáo chi tiết audit SHA-256 / pHash chống rò rỉ | P1 |
| **4.3** | Giao thức huấn luyện: 3 seeds độc lập, tối ưu trên validation, frozen test | `test_protocol.json` | `VERIFIED` | Bảng cấu hình learning rate, weight decay, optimizer | P1 |
| **4.4** | Cấu hình đối chứng và triệt tiêu: CLIP-only, Full Graph, No-edge, Random, Shuffled | `DB15K_Shuffled_Edges_v2_Review_Feedback.md` | `VERIFIED` | Phân biệt rõ no-edge của GAT (adapter) vs CLIP-only | **P0** |
| **4.5** | Thang đo đánh giá: Recall@1/5/10, MRR, Indirect MRR, Fidelity, Sparsity | `Project03_Multimodal_XAI.pdf` | `VERIFIED` | Công thức tính toán toán học chuẩn | P1 |
| **4.6** | Phòng chống rò rỉ: hash locking, kiểm soát overlap (v2 overlap = 0/0) | `DB15K_Shuffled_Edges_v2_Review_Feedback.md (Mục 3)` | `VERIFIED` | Công khai minh bạch giao thức test-aware exclusion | **P0** |
| **5.1.1** | Flickr30k CLIP baseline: Test MRR $\sim 0.7232$, R@1 $\sim 0.6120$ | `audit Flickr30k; 03_3_clip_baseline_retrieval.ipynb` | `VERIFIED` | Điền đầy đủ các giá trị R@5, R@10 chính xác từ audit log | P1 |
| **5.1.2** | Flickr30k CLIP + GAT 3 seeds: Mean MRR $\sim 0.8273 \pm 0.0020$ | `audit Flickr30k (Technical Audit Report)` | `VERIFIED` | Bảng số liệu chi tiết qua 3 seed 2026, 2027, 2028 | P1 |
| **5.1.3** | Flickr30k Ablation: No-edge adapter ($\sim 0.7810$) vs Random ($\sim 0.7450$) | `double-check/flickr30k_ablation.json` | `VERIFIED` | Khẳng định adapter đóng góp đáng kể, không quy hết cho đồ thị | **P0** |
| **5.1.4** | Phân tích định tính Flickr30k: 5 truy vấn thành công + 5 thất bại | `06_flickr30k_gat_xai_level1.ipynb; results/flickr30k/XAI` | `PARTIAL` | Trích xuất hình ảnh và đồ thị chú ý cho 10 trường hợp | P1 |
| **5.1.5** | Phát hiện chính và giới hạn Cấp độ 1 | `audit Flickr30k` | `VERIFIED` | Tóm tắt tỷ lệ đóng góp adapter vs. liên kết cạnh | P1 |
| **5.2.1** | DB15K AdaMF (seed 2027): Test MRR $0.414549$, R@10 $0.555596$ vs CLIP $0.2577$ | `DB15K_Shuffled_Edges_v2_Review_Feedback.md; test_metrics.csv` | `VERIFIED` | Điền các giá trị R@1, R@5 từ file metrics chính thức | P1 |
| **5.2.2** | Hiệu năng trên DB15K 3 seeds (mean $\sim 0.4133$) và FB15k-237 ($\sim 0.289$) | `test_protocol.json; 04b2_adamf_*.py` | `PARTIAL` | Tính độ lệch chuẩn SD của 3 seed DB15K; hoàn tất 3 seed FB15k-237 | **P0** |
| **5.2.3** | Phân tích Direct vs. Indirect trên DB15K: Indirect MRR $0.3164$ vs $0.0475$ | `DB15K_Shuffled_Edges_v2_Review_Feedback.md (Mục 2)` | `VERIFIED` | Bảng phân tích chi tiết bộ 20 truy vấn đánh giá | **P0** |
| **5.2.4** | DB15K Ablation: Full ($0.4145$) vs Shuffled v2 ($0.3156$) vs No-edges ($0.2577$) | `DB15K_Shuffled_Edges_v2_Review_Feedback.md (Mục 2)` | `VERIFIED` | Ghi nhận ablation chỉ có 1 seed (2027), không dùng mean $\pm$ SD | **P0** |
| **5.2.5** | Phát hiện chính và giới hạn Cấp độ 2 | `DB15K_Shuffled_Edges_v2_Review_Feedback.md` | `VERIFIED` | Nêu rõ giới hạn phương pháp test-aware exclusion | P1 |
| **5.3.1** | Kết quả sinh giải thích đường đi / đồ thị con | `results/*/XAI` | `PARTIAL` | Trích xuất đồ thị con cho các ví dụ tiêu biểu | P1 |
| **5.3.2** | Đánh giá định lượng XAI: Fidelity, Validity, Sparsity trên $\ge 10$ queries | `07_db15k_fb15k_adamf_xai_level2.ipynb` | `PARTIAL` | Tính giá trị trung bình định lượng và độ lệch chuẩn | **P0** |
| **5.3.3** | Phân tích chuyên sâu 5 thành công + 5 thất bại giải thích | `results/*/XAI/case_studies.json` | `PARTIAL` | Soạn thảo bảng đối chiếu 10 case study có ảnh minh chứng | P1 |
| **5.3.4** | Đánh giá trên Custom MMKG tiếng Việt | `dataset/mmkg_tv; 0. final/results/MMKG` | `NOT VERIFIED` | Chạy benchmark suy luận và xuất bảng kết quả retrieval | **P0** |
| **5.3.5** | Phát hiện chính và giới hạn Cấp độ 3 | `Project03_Multimodal_XAI.pdf` | `PARTIAL` | Tổng kết tính trung thực nhân quả vs ngụy biện hậu nghiệm | P1 |
| **5.4** | Đánh giá tổng hợp liên cấp độ: giải đáp trọn vẹn RQ1, RQ2, RQ3 | Toàn bộ kết quả thực nghiệm | `VERIFIED` | Tổng hợp xuyên suốt, tuyệt đối không so sánh cross-dataset | P1 |
| **6.1** | Thảo luận hiệu quả thực sự của truy vấn đồ thị theo loại truy vấn | Kết quả Chương 5 | `VERIFIED` | Phân tích bản chất hỗ trợ của đồ thị đối với CLIP | P2 |
| **6.2** | Đóng góp cấu trúc đồ thị: ngữ nghĩa quan hệ vs tiên nghiệm bậc vs adapter | Thí nghiệm triệt tiêu L1 \& L2 | `VERIFIED` | Bóc tách 3 thành phần cải thiện một cách khoa học | P1 |
| **6.3** | Độ tin cậy của giải thích đồ thị: tính trung thực mô hình vs tính chân lý | XAI case studies | `PARTIAL` | Phân tích bài học từ các ca dự đoán sai nhưng giải thích hợp lý | P1 |
| **6.4** | Giới hạn thực nghiệm: single-seed ablations, test-aware exclusion | Review feedback F01, F02 | `VERIFIED` | Thừa nhận minh bạch các rủi ro kỹ thuật | P1 |
| **6.5** | Hướng phát triển: strict test-blind, contrastive graph loss, video | Đề tài nghiên cứu mở rộng | `VERIFIED` | Đề xuất lộ trình nâng cấp hệ thống | P2 |
| **7.1** | Tóm tắt kết luận toàn diện theo 3 cấp độ nghiên cứu | Toàn bộ báo cáo | `VERIFIED` | Viết đoạn văn tổng kết học thuật liên tục | P1 |
| **7.2** | Đóng góp chính và các giới hạn còn lại | Toàn bộ báo cáo | `VERIFIED` | Đối chiếu với mục tiêu ban đầu | P1 |
| **7.3** | Hướng nghiên cứu tiếp theo | Lộ trình nghiên cứu | `VERIFIED` | Kế hoạch hoàn thiện trước ngày bảo vệ | P1 |
| **App A** | Bảng đối chiếu yêu cầu đề bài và tình trạng minh chứng | Đề bài \& Artifacts | `VERIFIED` | Cập nhật định kỳ khi có kết quả mới | P1 |
| **App B** | Cấu hình chi tiết siêu tham số và tài nguyên tính toán | Config files \& scripts | `VERIFIED` | Bổ sung chi tiết môi trường CUDA, thư viện | P2 |
| **App C** | Hướng dẫn tái lập thực nghiệm độc lập (Reproduction Guide) | Scripts \& manifests | `VERIFIED` | Kiểm thử chạy lại trên môi trường sạch | P1 |
| **App D** | Bảng phân công trách nhiệm và tỷ lệ đóng góp của thành viên | Bảng phân công nhóm | `PARTIAL` | Điền thông tin chính thức của các thành viên nhóm | P1 |

---

### Tóm tắt các công việc trọng tâm cần làm tiếp theo (Priority Tasks)

1. **[P0] Hoàn thiện bảng số liệu 3 seeds cho DB15K và FB15k-237:** Trích xuất độ lệch chuẩn SD của 3 seed DB15K (seeds 2027, 2028, 2029) và kiểm tra trạng thái chạy đủ 3 seed của FB15k-237 để thay thế các placeholder trong Bảng 5.3 và 5.4.
2. **[P0] Trích lọc và chuẩn hóa Bộ 20 truy vấn đánh giá Cấp độ 2:** Cố định danh sách 10 câu truy vấn trực tiếp và 10 câu truy vấn gián tiếp kèm theo thứ hạng so sánh giữa CLIP và AdaMF từ file `test_query_ranks.csv` của DB15K.
3. **[P0] Đánh giá định lượng XAI trên $\ge 10$ queries:** Thu thập số liệu đo lường can thiệp nhiễu xóa cạnh để tính toán các giá trị trung bình $\text{Fidelity}^-$, $\text{Validity}$ và $\text{Sparsity}$ điền vào Bảng 5.5.
4. **[P0] Triển khai benchmark trên Custom MMKG tiếng Việt (Cấp độ 3):** Hoàn thiện tiền xử lý và chạy suy luận mô hình AdaMF tốt nhất trên tập dữ liệu `dataset/mmkg_tv`, tạo artifact kiểm chứng tại `0. final/results/MMKG`.
5. **[P1] Trích xuất hình ảnh định tính cho 10 ca Flickr30k và 10 ca MMKG:** Lưu trữ các ảnh và sơ đồ đồ thị con vào thư mục `report/images/` để chèn vào các vị trí hình dự kiến trong báo cáo.
