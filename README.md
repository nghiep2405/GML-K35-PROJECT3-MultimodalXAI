# Project 03 — Multimodal XAI

Đề tài: **Truy vấn đa phương thức có giải thích bằng đồ thị (Multimodal Retrieval with Graph-based Explainability)**  
Môn học: **Học máy với dữ liệu đồ thị (Machine Learning with Graphs)** — Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM  
Nhóm thực hiện: **Nhóm 02** (5 thành viên)  
Đề bài chi tiết nằm trong [Project03_Multimodal_XAI.pdf](Project03_Multimodal_XAI.pdf).

---

Xem [cấu trúc repo và cách chạy từng pipeline](structure_instruction__repo.md) để cài package, chạy notebook và kiểm thử.

## 1. Cấu trúc thư mục dự án

| Thư mục | Nội dung |
| :--- | :--- |
| [doc/](doc/) | Ghi chú, tài liệu hướng dẫn tiền xử lý và đặc tả bài toán của đồ án. |
| [report/](report/) | Toàn bộ mã nguồn LaTeX, hình ảnh và tài liệu tham khảo cho Báo cáo kỹ thuật cuối kỳ. |
| [slide/](slide/) | Mã nguồn LaTeX Beamer và script biên dịch bộ slide báo cáo chuẩn 16:9 (17 slides). |
| [demo/](demo/) | Kịch bản chi tiết cho video demo dự phòng ($\le$ 5 phút) và hướng dẫn live demo. |
| [src/](src/) | Toàn bộ mã nguồn triển khai, pipeline huấn luyện, trích xuất đặc trưng và đánh giá. |

---

## 2. Hướng dẫn biên dịch Báo cáo kỹ thuật và Slide

### 2.1. Biên dịch báo cáo kỹ thuật (Report)
Cần cài đặt bộ LaTeX có `pdfLaTeX` và `BibTeX`. Từ thư mục gốc của repository, chạy:

```bash
./report/compile.sh
```

Hoặc chạy trực tiếp từ thư mục `report/`:
```bash
cd report && ./compile.sh
```
Mỗi lần chạy sẽ tạo một thư mục riêng theo thời gian trong `report/report_compiled/` (ví dụ: `report/report_compiled/2026-11-15_17-44-38/main.pdf`). Thư mục này đã được cấu hình trong `.gitignore` để tránh đưa các file biên dịch nặng lên GitHub.

### 2.2. Biên dịch slide báo cáo (Slide Beamer)
Bộ slide được thiết kế theo chuẩn LaTeX Beamer widescreen (tỉ lệ 16:9), đồng bộ nhận diện màu `HCMUSBlue` của trường. Để biên dịch, chạy:

```bash
cd slide && ./compile_slide.sh
```
Script sẽ tự động tạo thư mục lưu trữ theo timestamp trong `slide/slide_compiled/` (ví dụ: `slide/slide_compiled/2026-11-15_18-29-11/main_slide.pdf`) và sao chép bản PDF mới nhất ra `slide/main_slide.pdf` để tiện xem trước. Toàn bộ thư mục `slide/slide_compiled/` và file `slide/main_slide.pdf` đã được thiết lập trong `.gitignore`.

---

## 3. Chiến lược phân chia thuyết minh và demo (Khung thời gian 20 phút)

Theo quy định tại **Mục 8 & 9** của đề tài (*Project03_Multimodal_XAI.pdf*):
- Thời lượng trình bày tối đa: **20 phút** bao gồm cả Slide lý thuyết và Demo chạy trực tiếp.
- Sau 20 phút trình bày là phần **Giảng viên vấn đáp từng thành viên độc lập**.
- Giảng viên có quyền chất vấn bất kỳ ai về dữ liệu, đồ thị, mô hình, mã nguồn, kết quả, xAI và phòng chống data leakage để chấm điểm cá nhân.

### 3.1. Phân tích: Nên chia bao nhiêu người thuyết trình là hợp lý?

| Số lượng người nói | Đánh giá ưu điểm | Hạn chế và rủi ro | Mức độ khuyến nghị |
| :---: | :--- | :--- | :---: |
| **1 người duy nhất** | Mạch nói rất liền mạch, không tốn thời gian đổi người, dễ căn chỉnh giờ. | Các thành viên khác bị thụ động; giảng viên dễ có tâm lý khắt khe hơn khi vấn đáp các bạn còn lại. | Không khuyến nghị |
| **Cả 5 người cùng nói** | Mọi thành viên đều được nói trước hội đồng trong phần slide. | Với 15 phút slide mà đổi người 4 lần (mất 15--20s mỗi lần chuyển mic, chào hỏi, lấy lại nhịp), mạch lập luận bị đứt gãy nghiêm trọng, người nói sau bị áp lực chạy đua thời gian, **rất dễ bị quá giờ (lố 20 phút)**. | Chỉ dùng khi Thầy/Cô bắt buộc |
| **3 người nói chính + 2 người phụ trách demo & kỹ thuật** | **Cân bằng hoàn hảo**: Mạch thuyết trình chỉ đổi người đúng 2 lần tại các ranh giới tự nhiên giữa các cấp độ (Level 1 $\rightarrow$ Level 2 $\rightarrow$ Level 3); mỗi người có đủ 4--5.5 phút để trình bày sâu sắc bản chất khoa học; 2 bạn còn lại làm chủ hoàn toàn phần demo và kỹ thuật dữ liệu. | Cần tập dượt phối hợp nhịp nhàng giữa người thuyết minh và người thao tác máy tính. | **KHUYẾN NGHỊ TỐI ƯU** |

---

### 3.2. Kế hoạch phân bổ chi tiết theo phương án tối ưu (Khuyến nghị)

Tổng thời gian: **20 phút** = **15 phút Slide** + **4.5 phút Demo** + **0.5 phút chuyển giao**.

```
[ TỔNG THỜI GIAN TRÌNH BÀY: 20 PHÚT ]
│
├── GIAI ĐOẠN 1: SLIDE BÁO CÁO HỌC THUẬT (00:00 - 15:00, 15 phút)
│   ├── Người 1: Chung Gia Khánh (00:00 - 05:30, 5.5 phút) -- Slide 1 đến 6
│   │   • Mở đầu, Bối cảnh nghiên cứu và Phát biểu bài toán.
│   │   • Kiến trúc hệ thống 3 tầng và 3 câu hỏi nghiên cứu (RQ1, RQ2, RQ3).
│   │   • Cấp độ 1: CLIP Baseline, mạng GAT trên Flickr30k Entities.
│   │   • Kết quả Cấp độ 1 và phân tích triệt tiêu (Ablation: Adapter effect vs. Cạnh đồ thị).
│   │
│   ├── Người 2: Trần Gia Hiển (05:30 - 11:00, 5.5 phút) -- Slide 7 đến 10
│   │   • Động lực chuyển tiếp lên MMKG đa quan hệ (DB15K, FB15k-237).
│   │   • Cơ chế dung hợp đặc trưng thích nghi AdaMF (gating network thị giác--văn bản).
│   │   • Đột phá trên truy vấn gián tiếp (Indirect query: Indirect MRR tăng gấp 6.6 lần).
│   │   • Kiểm thử hoán đổi cạnh bảo toàn bậc (Shuffled Edges v1 vs. Shuffled Edges v2 overlap 0/0).
│   │
│   └── Người 3: Nguyễn An Nghiệp (11:00 - 15:00, 4.0 phút) -- Slide 11 đến 14, 16, 17
│       • Cấp độ 3: Khung giải thích xAI trích xuất đồ thị con nhân quả.
│       • Đo lường định lượng 3 chỉ số: Fidelity, Validity, Sparsity qua can thiệp xóa cạnh.
│       • Khảo sát chuyển giao trên tập dữ liệu MMKG tiếng Việt tự xây.
│       • Bốn trụ cột phòng chống data leakage (Artifact Hash Locking, Split v2 pHash, Frozen Test).
│       • Kết luận 4 đóng góp chính và hướng phát triển tiếp theo.
│
├── GIAI ĐOẠN 2: THỰC NGHIỆM DEMO TRỰC TIẾP & VIDEO (15:00 - 19:30, 4.5 phút)
│   └── Người 4 & 5: Lê Hoàng Mỹ Hạ & Hàn Vũ Phương Uyên -- Phụ trách máy tính và thuyết minh Demo
│       • 15:00 - 15:35: Xác thực mã băm SHA-256 của checkpoint và dataset contract trên giao diện.
│       • 15:35 - 16:30: Demo Cấp độ 1 (Flickr30k) -- Bật attention weights trên bounding box ảnh.
│       • 16:30 - 17:45: Demo Cấp độ 2 (DB15K) -- So sánh Direct query vs. Indirect query (vụ Eiffel Tower).
│       • 17:45 - 19:00: Demo Cấp độ 3 (xAI) -- Trực quan hóa subgraph và kiểm thử xóa cạnh (Fidelity Drop).
│       • 19:00 - 19:30: Minh họa 1 failure case và cấu trúc Data Card dữ liệu tiếng Việt.
│
└── GIAI ĐOẠN 3: KẾT THÚC & VẤN ĐÁP CÁ NHÂN (19:30 - 20:00+)
    └── Toàn bộ 5 thành viên sẵn sàng trước bục trả lời câu hỏi của Giảng viên.
```

---

### 3.3. Phân công vai trò cụ thể khi bước vào phần Vấn đáp (Q&A)

Để đạt điểm tối đa ở phần câu hỏi cá nhân của Thầy/Cô, các thành viên phân định mũi nhọn chuyên sâu như sau:

1. **Chung Gia Khánh (Trưởng nhóm):**
   - Phụ trách trả lời: Kiến trúc tổng thể, quy chuẩn phương pháp luận nghiên cứu, thiết kế thử nghiệm tổng thể, cơ chế frozen test, phân lập adapter effect ở Level 1 và tổng kết báo cáo.
2. **Lê Hoàng Mỹ Hạ:**
   - Phụ trách trả lời: Quy trình tiền xử lý dữ liệu, kiểm soát rò rỉ hình ảnh qua phân cụm pHash/SHA-256 trên Visual Genome (*split_v2*), trích xuất ma trận đặc trưng OpenCLIP ViT-B/32 đóng băng và lưu trữ tensor nhị phân.
3. **Hàn Vũ Phương Uyên:**
   - Phụ trách trả lời: Xây dựng đồ thị cảnh (scene graph), cấu trúc đỉnh/cạnh Flickr30k Entities, kiến trúc mạng GAT cục bộ, cơ chế lan truyền thông điệp self-attention và phân tích các trường hợp thành công/thất bại Level 1.
4. **Trần Gia Hiển:**
   - Phụ trách trả lời: Toán học của cơ chế dung hợp đặc trưng AdaMF (gating network), cấu trúc bộ ba đa phương thức DB15K và FB15k-237, giao thức hoán đổi cạnh Shuffled Edges v2 và kỹ thuật loại trừ test-aware exclusion bảo đảm overlap 0/0.
5. **Nguyễn An Nghiệp:**
   - Phụ trách trả lời: Thuật toán trích xuất đồ thị con giải thích xAI, công thức toán và mã nguồn đo lường Fidelity, Validity, Sparsity, kiểm thử can thiệp làm nhiễu đồ thị, cùng quy trình thu thập metadata Data Card cho MMKG tiếng Việt.

---

### 3.4. Kịch bản dự phòng (Nếu Giảng viên yêu cầu bắt buộc cả 5 người phải nói trong 20 phút)

Nếu tại buổi bảo vệ, Thầy/Cô có yêu cầu bắt buộc mọi thành viên đều phải tham gia thuyết trình phần slide, nhóm chuyển sang kịch bản chia 5 phần đều đặn:

- **Thành viên 1 (Khánh, 3.5 phút):** Mở đầu, Bối cảnh, Phát biểu bài toán, Mục tiêu nghiên cứu và Kiến trúc tổng thể (Slide 1--4).
- **Thành viên 2 (Hạ / Uyên, 3.5 phút):** Cấp độ 1: Mô hình cơ sở CLIP, Mạng GAT, Kết quả 3 seeds và Ablation adapter (Slide 5--6).
- **Thành viên 3 (Hiển, 4.0 phút):** Cấp độ 2: Đồ thị tri thức MMKG, Cơ chế AdaMF, Đột phá Indirect query và Shuffled Edges v2 (Slide 7--10).
- **Thành viên 4 (Nghiệp, 3.5 phút):** Cấp độ 3: Khung giải thích xAI, Định lượng Fidelity/Validity/Sparsity và MMKG tiếng Việt (Slide 11--13).
- **Thành viên 5 (Uyên / Hạ, 5.0 phút):** Quy chuẩn phòng chống data leakage (Slide 14) + Điều khiển và thuyết minh Video/Live Demo 5 phút (Slide 15) + Kết luận và Bảng đóng góp (Slide 16--17).

---

### 3.5. Quy tắc chuyển giao lời thoại mượt mà (Tránh mất thời gian)
- **Không chào hỏi lặp lại:** Khi chuyển người, thành viên tiếp theo vào thẳng nội dung, ví dụ: *"Tiếp theo, em xin phép đại diện nhóm trình bày về phương pháp dung hợp AdaMF ở Cấp độ 2..."*.
- **Người điều khiển slide cố định:** Một thành viên cầm clicker hoặc ngồi máy tính chuyển slide xuyên suốt để người thuyết trình tập trung nhìn vào hội đồng, không bị gián đoạn thao tác.
- **Canh thời gian chặt chẽ:** Thành viên hỗ trợ dùng đồng hồ đếm ngược đặt ở vị trí người nói dễ nhìn thấy nhất (nhắc mốc 10 phút, 15 phút, 18 phút).
