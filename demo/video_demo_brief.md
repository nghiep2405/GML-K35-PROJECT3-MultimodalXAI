# KỊCH BẢN VÀ HƯỚNG DẪN THỰC HIỆN VIDEO DEMO

## THÔNG TIN CHUNG
- **Đề tài:** PROJECT 03: TRUY VẤN ĐA PHƯƠNG THỨC CÓ GIẢI THÍCH BẰNG ĐỒ THỊ
- **Môn học:** Học máy với dữ liệu đồ thị -- Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM
- **Nhóm thực hiện:** Nhóm 02
- **Thời lượng video:** Tối đa 5:00 (chuẩn bị video dự phòng theo quy định đề bài tại Mục 8 & 9, phục vụ chiếu hoặc chạy trực tiếp trong 20 phút bảo vệ).
- **Mục tiêu demo:** Minh họa trực quan khả năng truy vấn văn bản tìm ảnh, so sánh hiệu năng giữa baseline CLIP thuần và mô hình đồ thị (GAT / AdaMF), chứng minh năng lực vượt trội trên truy vấn gián tiếp, và trực quan hóa giải thích xAI có kiểm chứng tính trung thực nhân quả (Fidelity).

---

## CẤU TRÚC PHÂN ĐOẠN VIDEO DEMO (TỔNG CỘNG 5 PHÚT)

| Phân đoạn | Thời lượng | Nội dung trọng tâm | Thao tác trên màn hình |
| :--- | :---: | :--- | :--- |
| **Phần 1: Giới thiệu & Xác thực môi trường** | 0:00 - 0:35 (35s) | Giới thiệu giao diện demo, load checkpoint đóng băng, xác thực mã băm SHA-256 chống data leakage | Mở giao diện Notebook / Web UI, chạy lệnh kiểm tra hash `test_protocol.json` và checkpoint |
| **Phần 2: Demo Cấp độ 1 (CLIP + GAT)** | 0:35 - 0:35 (60s) | Thử nghiệm trên Flickr30k Entities: so sánh kết quả CLIP vs. GAT, xem attention weight trên bounding box | Nhập câu truy vấn trực tiếp mô tả đối tượng, hiển thị top-5 ảnh và trọng số chú ý cục bộ |
| **Phần 3: Demo Cấp độ 2 (AdaMF trên MMKG)** | 1:35 - 2:50 (75s) | Điểm nhấn chính: so sánh truy vấn trực tiếp vs. truy vấn gián tiếp trên DB15K | Nhập Direct query (CLIP tốt) và Indirect query (CLIP rớt hạng, AdaMF đưa lên top-1) |
| **Phần 4: Demo Cấp độ 3 (xAI & Fidelity Test)** | 2:50 - 4:05 (75s) | Trích xuất đồ thị con giải thích và thực hiện kiểm thử can thiệp xóa cạnh (Fidelity perturbation) | Bấm trích xuất đường đi giải thích; chạy thử xóa cạnh giải thích $\rightarrow$ rank tụt; xóa cạnh ngẫu nhiên $\rightarrow$ rank giữ nguyên |
| **Phần 5: Phân tích thất bại & Dữ liệu tiếng Việt** | 4:05 - 4:50 (45s) | Minh họa 1 trường hợp thất bại để chỉ rõ giới hạn; hiển thị mẫu đồ thị MMKG tiếng Việt | Mở case study thất bại định tính; chiếu nhanh cấu trúc thư mục Data Card tiếng Việt |
| **Phần 6: Kết luận & Chuyển tiếp vấn đáp** | 4:50 - 5:00 (10s) | Tóm tắt các minh chứng kỹ thuật đã đạt được | Hiện slide kết thúc demo, sẵn sàng cho giảng viên đặt câu hỏi |

---

## KỊCH BẢN CHI TIẾT TỪNG PHÂN ĐOẠN

### Phân đoạn 1: Giới thiệu hệ thống và xác thực môi trường (0:00 - 0:35)
- **Mục tiêu:** Tạo ấn tượng chuyên nghiệp, khẳng định ngay tính trung thực học thuật và quy chuẩn chống data leakage.
- **Thao tác màn hình:**
  1. Mở màn hình giao diện demo (Jupyter Notebook / Streamlit app).
  2. Hiển thị thông tin phiên bản mô hình, seed ngẫu nhiên (`2026`, `2027`), và chạy hàm kiểm tra mã băm SHA-256 của checkpoint và file danh mục dữ liệu.
- **Lời thuyết minh mẫu:**
  > *"Kính chào quý thầy cô. Đây là phần video demo cho Project 03: Truy vấn đa phương thức có giải thích bằng đồ thị của Nhóm 02. Hệ thống demo được đóng gói độc lập, vận hành trên các checkpoint mô hình đã đóng băng hoàn toàn. Trước khi truy vấn, hệ thống thực hiện xác thực mã băm SHA-256 đối với bộ đặc trưng và test set để bảo đảm không xảy ra rò rỉ dữ liệu qua bất kỳ bước tinh chỉnh nào."*

---

### Phân đoạn 2: Demo Cấp độ 1 - Truy vấn tăng cường đồ thị GAT (0:35 - 1:35)
- **Mục tiêu:** Minh họa cách thức GAT cải thiện điểm số so với baseline CLIP thuần và giải thích dựa trên attention weight cục bộ.
- **Thao tác màn hình:**
  1. Chọn tập dữ liệu **Flickr30k Entities**.
  2. Nhập câu truy vấn kiểm thử (ví dụ: *"a girl in blue shirt jumping into water"*).
  3. Bấm **Chạy truy vấn (Run Retrieval)**:
     - Cột trái: Kết quả xếp hạng của mô hình baseline CLIP thuần túy (ảnh đúng đứng ở rank 4 hoặc 5).
     - Cột phải: Kết quả xếp hạng sau khi rerank bằng CLIP + GAT ($\alpha = 0.6$) (ảnh đúng được đẩy lên rank 1).
  4. Bật chế độ hiển thị giải thích Cấp độ 1: Trực quan hóa các bounding box đối tượng trong ảnh kèm trọng số attention lớn nhất giữa cụm từ và đối tượng.
- **Lời thuyết minh mẫu:**
  > *"Ở Cấp độ 1, với câu truy vấn mô tả trực tiếp đối tượng trên Flickr30k, CLIP thuần túy đã tìm được ảnh nhưng chỉ xếp ở vị trí thứ 4 do nhầm lẫn nền nước với các bối cảnh khác. Khi kích hoạt mạng GAT, thông tin cấu trúc giữa thực thể 'girl' và hành động 'jumping' được tổng hợp, giúp đẩy ảnh chính xác lên vị trí Top-1. Trọng số attention cho thấy mô hình tập trung cao nhất vào vùng đối tượng cô gái và liên kết hành động liên quan."*

---

### Phân đoạn 3: Demo Cấp độ 2 - Đột phá trên truy vấn gián tiếp với AdaMF (1:35 - 2:50)
- **Mục tiêu:** Trình diễn năng lực cốt lõi nhất của đồ án: MMKG giải quyết bài toán truy vấn gián tiếp mà mô hình thị giác--ngôn ngữ thuần túy bất khả thi.
- **Thao tác màn hình:**
  1. Chuyển sang cơ sở tri thức **DB15K** (hoặc FB15k-237).
  2. **Thử nghiệm 1 (Direct Query):** Nhập *"Eiffel Tower in Paris"*. Cả CLIP và AdaMF đều đưa thực thể đích vào Top-1.
  3. **Thử nghiệm 2 (Indirect Query - Trọng tâm):** Nhập câu truy vấn gián tiếp: *"The monument designed by Gustave Eiffel for the 1889 World's Fair"*.
     - Kết quả CLIP baseline: Thực thể đích bị trôi xuống rank $>50$ (do trong ảnh chỉ có hình tháp, không có chữ Gustave Eiffel hay năm 1889).
     - Kết quả AdaMF Hybrid ($\alpha = 0.2$): Nhờ khai thác bộ ba quan hệ `(Gustave_Eiffel, designed, Eiffel_Tower)` và `(Eiffel_Tower, built_for, 1889_Exposition)`, mô hình suy luận bắc cầu và đưa đúng thực thể đích lên vị trí Top-1.
  4. Hiển thị trực quan trọng số cổng thích nghi AdaMF: biểu diễn tỷ lệ đóng góp giữa văn bản ($\gamma_T$) và hình ảnh ($\gamma_V$).
- **Lời thuyết minh mẫu:**
  > *"Điểm nhấn học thuật lớn nhất nằm ở Cấp độ 2 khi xử lý các truy vấn gián tiếp. Khi người dùng tìm kiếm 'Công trình do Gustave Eiffel thiết kế cho Hội chợ Thế giới 1889', CLIP hoàn toàn thất bại vì đặc trưng thị giác bề mặt không thể nhìn thấy tên kiến trúc sư hay mốc lịch sử. Nhưng trên đồ thị MMKG, mô hình AdaMF đã kích hoạt đường truyền tri thức đa chặng, giúp thực thể đích vươn từ vị trí ngoài Top-50 lên thẳng vị trí số 1. Trọng số cổng AdaMF tự thích nghi gán tỷ trọng tri thức quan hệ cao để dẫn đường chính xác."*

---

### Phân đoạn 4: Demo Cấp độ 3 - Khung giải thích xAI & Kiểm chứng Fidelity (2:50 - 4:05)
- **Mục tiêu:** Chứng minh giải thích không phải là bịa đặt hậu nghiệm mà có tính trung thực nhân quả thực sự đối với thuật toán chấm điểm.
- **Thao tác màn hình:**
  1. Bấm nút **Trích xuất giải thích (Extract Explanation Subgraph)** cho kết quả Top-1 vừa tìm được ở Phân đoạn 3.
  2. Hệ thống hiển thị đồ thị con giải thích tối giản:
     - Gồm thực thể nguồn, thực thể trung gian, thực thể đích và các cạnh quan hệ liên kết.
     - Hiển thị chỉ số Sparsity ($88\%$, rất cô đọng) và Validity ($1.00$).
  3. **Thực hiện kiểm thử can thiệp (Perturbation Test / Fidelity Check):**
     - Bấm nút **Xóa cạnh giải thích (Mask Explanation Edges)**: Điểm số kết hợp tụt giảm mạnh, thứ hạng của thực thể lập tức rơi từ Top-1 xuống Top-45 $\rightarrow$ Chứng minh Fidelity đạt mức cao.
     - Bấm nút **Xóa cạnh ngẫu nhiên đối chứng (Mask Random Edges)**: Thứ hạng của thực thể đích vẫn được bảo toàn ở Top-1 $\rightarrow$ Chứng minh sự sụt giảm ở trên không phải do ngẫu nhiên mà bắt nguồn từ chính các cạnh giải thích.
- **Lời thuyết minh mẫu:**
  > *"Tại Cấp độ 3, chúng tôi không chỉ đưa ra kết quả mà còn mở chiếc hộp đen ra quyết định. Hệ thống trích xuất đường đi suy luận tối giản với độ thưa đạt trên 85%. Để chứng minh tính trung thực nhân quả (Fidelity), chúng tôi tiến hành can thiệp: khi xóa bỏ các cạnh giải thích, thứ hạng của thực thể rơi ngay lập tức từ Top-1 xuống Top-45. Ngược lại, khi xóa một tập cạnh ngẫu nhiên cùng kích thước, thứ hạng vẫn giữ vững ở Top-1. Điều này chứng minh giải thích sinh ra phản ánh đúng cơ chế tính điểm bên trong của mô hình."*

---

### Phân đoạn 5: Phân tích trường hợp thất bại & Dữ liệu tự xây tiếng Việt (4:05 - 4:50)
- **Mục tiêu:** Thể hiện sự khách quan, trung thực khoa học, nêu rõ các giới hạn còn lại đúng theo tiêu chuẩn học thuật.
- **Thao tác màn hình:**
  1. Chọn một trường hợp truy vấn thất bại (Failure case study): Ví dụ một truy vấn có nhiều thực thể đồng dạng hoặc quan hệ gây nhiễu, phân tích vì sao mô hình đưa ra xếp hạng chưa chính xác.
  2. Mở tab **Custom MMKG tiếng Việt**: Chiếu cấu trúc thư mục dữ liệu đã thu thập tại `dataset/mmkg_tv`, file Data Card minh chứng nguồn ảnh và bảng bộ ba quan hệ.
- **Lời thuyết minh mẫu:**
  > *"Tính trung thực học thuật cũng được thể hiện qua việc phân tích các trường hợp thất bại. Ở ví dụ này, khi câu truy vấn có các thực thể mang liên kết dày đặc với nhau, mô hình bị nhiễu do ưu tiên đỉnh có bậc cao hơn ngữ nghĩa thực sự. Cuối cùng, nhóm trình bày tiến độ của tập MMKG tiếng Việt miền đặc thù, đã hoàn tất thu thập ảnh và xây dựng bảng metadata, định vị rõ ranh giới các artifact đã kiểm chứng và các phần đang hoàn thiện."*

---

### Phân đoạn 6: Kết luận và chuyển tiếp vấn đáp (4:50 - 5:00)
- **Mục tiêu:** Kết thúc video gọn gàng, đúng thời lượng quy định.
- **Thao tác màn hình:**
  - Hiển thị màn hình tổng kết các chỉ số chính (Flickr30k MRR 0.827, DB15K Indirect MRR 0.316, Fidelity verified) và logo nhóm.
- **Lời thuyết minh mẫu:**
  > *"Video demo của Nhóm 02 đến đây là kết thúc. Chúng em xin cảm ơn quý thầy cô và sẵn sàng bước vào phần vấn đáp chi tiết."*

---

## CHECKLIST KỸ THUẬT VÀ PHÒNG NGỪA RỦI RO KHI DEMO

### 1. Chuẩn bị trước buổi bảo vệ
- [ ] Chạy sẵn môi trường ảo: `source venv/bin/activate`.
- [ ] Khởi chạy notebook / giao diện demo sẵn trên máy cục bộ, kiểm tra cổng mạng (`localhost:8501` hoặc `localhost:8888`).
- [ ] Cache sẵn đặc trưng CLIP cho các truy vấn mẫu để phản hồi demo tức thì dưới 1 giây.
- [ ] Chuẩn bị sẵn file video quay màn hình chất lượng cao (1080p, âm thanh rõ ràng, thời lượng $\le$ 4:50) lưu trên Desktop và USB dự phòng trường hợp máy tính gặp sự cố.

### 2. Các câu hỏi vấn đáp tiềm năng của giảng viên khi xem demo
1. **"Làm sao chứng minh kết quả demo không bị overfit hay rò rỉ dữ liệu (data leakage)?"**
   - *Cách trả lời:* Mô hình chạy trên tập test cố định hoàn toàn độc lập với tập train. Checkpoint và ma trận test được khóa mã băm SHA-256 bất biến. Trọng số $\alpha$ được tối ưu trên validation set trước khi khóa cố định sang test set.
2. **"Tại sao khi xóa cạnh giải thích thì điểm lại giảm, cơ chế tính điểm Fidelity là gì?"**
   - *Cách trả lời:* Điểm xếp hạng phụ thuộc vào hàm chấm điểm bộ ba của AdaMF và lan truyền GAT. Khi cạnh giải thích bị mask, đường dẫn tri thức bị đứt đoạn khiến $S_{\text{graph}}$ tụt mạnh, dẫn đến điểm lai ghép $S_{\text{hybrid}}$ giảm và thứ hạng rớt xuống.
3. **"Khác biệt cơ bản giữa Shuffled Edges v1 và v2 là gì?"**
   - *Cách trả lời:* Bản v1 bị trùng 171 cạnh với test set. Bản v2 đã loại bỏ triệt để 100% cạnh trùng (overlap 0/0) bằng phương pháp test-aware exclusion, đảm bảo tính trong sạch tuyệt đối cho phép thử triệt tiêu quan hệ.
