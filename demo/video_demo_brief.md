# Kịch bản video demo Project 03 (Nhóm 02)

Đề, mục 8: "Demo chạy trực tiếp khi vấn đáp, chuẩn bị video dự phòng không quá 5 phút" và "Notebook hoặc giao diện nhỏ: nhập truy vấn,
trả về top-k kết quả và hiển thị quan hệ/đường đi giải thích". Video là bản quay sẵn của phần demo trực tiếp, nên quay đúng giao diện
dưới đây. Mục tiêu: 4:45.

Mỗi lần quay phải thấy đủ ba việc: **nhập truy vấn → top-k ảnh → đồ thị giải thích**, và trả lời được ba câu hỏi của đề:
kết quả nào phù hợp nhất, tri thức quan hệ giúp ở đâu (nhất là truy vấn gián tiếp), vì sao mô hình trả về kết quả này.

## 1. Chuẩn bị

| Cấp | Notebook (thư mục `demo/`) | Dữ liệu, mô hình |
|---|---|---|
| 1 | `demo_cap1_visual_genome.ipynb` | Visual Genome, CLIP + GAT (checkpoint chính đã khóa) |
| 2 | `demo_cap2_db15k_fb15k.ipynb` | DB15K và FB15K-237-IMG, CLIP + AdaMF |
| 3 | `demo_cap3_viheritage.ipynb` | ViHeritage-MMKG, CLIP + GAT và CLIP + AdaMF |

1. Notebook cấp 2 cần bản `src/` mới: hoặc chép thư mục `src` của repo vào `MyDrive/MultimodalXAI/src`, hoặc push nhánh `demo`
   lên GitHub để notebook tự tải (bỏ qua file dữ liệu lớn). Notebook cấp 1 và cấp 3 đã chứa sẵn toàn bộ mã.
2. Mở từng notebook trên Colab (GPU), Runtime → Run all. Ô cuối in một link `https://….gradio.live`; mở ở tab riêng, toàn màn hình.
3. Chọn trước một lần mọi ca sẽ quay. Kết quả được lưu đệm, khi quay sẽ hiện ngay.
4. Quay 1920×1080, trình duyệt phóng 100% và bấm F11 (toàn màn hình) để thấy cả ô chọn ca lẫn toàn bộ trang kết quả.
   Ẩn thanh dấu trang và email tài khoản. Quay từng đoạn rồi ghép.
5. Chỉ đọc số có trên màn hình hoặc trong báo cáo. Chỗ ghi **[đọc trên màn hình]** là số trang web tính tại chỗ.
6. Quay xong: Runtime → Disconnect để link ngừng hoạt động.

**Cách đọc màn hình (giống nhau ở cả ba cấp):**
- Dòng trên cùng: câu truy vấn, neo và quan hệ (nếu có), và "Ảnh đúng: CLIP hạng a → hạng b".
- Bên trái: ảnh đang xem, ✓ nếu là ảnh đúng. Bên phải: đồ thị trả lời "Vì sao trả về X?".
- Màu trên đồ thị:
  - nút xanh ngọc là neo, nút cam là chính bức ảnh trả về, nút xám là nút khác;
  - cạnh xanh dương là phần tử của giải thích S, dày theo Δ (số trong ngoặc);
  - cạnh chấm xanh nhạt là đường nối về neo;
  - cạnh xám nét đứt là quan hệ được hỏi.
- Nút **Bỏ giải thích S** làm mờ các cạnh xanh và hiện hạng của ảnh sau khi bỏ S, so với bỏ cùng số phần tử ngẫu nhiên.

## 2. Kịch bản

### Đoạn 0 · 0:00 đến 0:15 · Mở đầu
- **Màn hình:** tab cấp 1.
- **Lời:** "Đây là demo Project 03 của Nhóm 02: truy vấn đa phương thức có giải thích bằng đồ thị. Hệ thống nhận một câu truy vấn,
  trả về top-k ảnh, xếp hạng lại bằng đồ thị và cho thấy quan hệ nào đã đưa ảnh lên. Nhóm demo ba cấp trên cùng một giao diện."

### Đoạn 1 · 0:15 đến 1:15 · Cấp 1, Visual Genome
- **Thao tác:**
  1. Chọn ca `[thành công] Guy holding up peace sign (CLIP 15 → 1)`.
  2. Rê chuột lên cạnh dày nhất.
  3. Bấm **Bỏ giải thích S**.
- **Màn hình:**
  - "Ảnh đúng: CLIP hạng 15 → CLIP + GAT hạng 1".
  - Đồ thị: bức ảnh ở giữa, cạnh dày nhất là "chứa ×5" tới peace sign.
  - Sau khi bấm: ảnh xuống hạng 4.
- **Lời:** "Ở cấp 1, CLIP lấy 50 ảnh ứng viên, GAT chạy trên scene graph của từng ảnh để chấm điểm đồ thị; điểm cuối gồm 0.6 phần CLIP
  và 0.4 phần đồ thị. Với câu này, CLIP xếp ảnh đúng hạng 15, CLIP + GAT đưa lên hạng 1. Đồ thị trả lời vì sao: ảnh chứa năm
  peace sign, đó là phần đóng góp lớn nhất. Bỏ các cạnh xanh này, ảnh rơi xuống hạng 4, còn bỏ ngẫu nhiên cùng số cạnh thì
  **[đọc trên màn hình]**. Vậy giải thích lấy từ chính hàm xếp hạng."
- **Thao tác (ca thất bại, khoảng 15 giây):** chọn `[thất bại, đồ thị làm hại] A man on the road (CLIP 1 → 3)`.
- **Lời:** "Ở ca thất bại, ảnh sai cũng chứa road và man nên được cộng điểm. Mô hình thưởng cho vật có mặt nhưng chưa hiểu quan hệ
  'on'. Ablation cũng cho thấy bỏ cạnh vẫn giữ khoảng 77% mức tăng."

### Đoạn 2 · 1:15 đến 2:35 · Cấp 2, DB15K
- **Thao tác:**
  1. Sang tab cấp 2, chọn ca đầu tiên `[thành công] … influenced by relation points to Richard Dawkins. (CLIP 55 → 1)`.
  2. Bấm **Bỏ giải thích S**.
- **Màn hình:**
  - Neo: Richard Dawkins · Quan hệ: influenced by.
  - "Ảnh đúng: CLIP hạng 55 → CLIP + AdaMF hạng 1".
  - Đồ thị: cạnh *influenced* (0.198) nối Christopher Hitchens thẳng với neo, và một bộ ba thứ hai (0.129).
- **Lời:** "Cấp 2 dùng đồ thị tri thức đa phương thức với AdaMF. Truy vấn gián tiếp cho biết thực thể neo và quan hệ. Ảnh của
  Christopher Hitchens không cho thấy quan hệ với Richard Dawkins nên CLIP chỉ xếp hạng 55; AdaMF đưa lên hạng 1. Lý do nằm ngay
  trên đồ thị: bộ ba Hitchens influenced Dawkins nối thẳng với neo. Bỏ hai bộ ba này, ảnh **[đọc trên màn hình]**; bỏ hai bộ ba
  ngẫu nhiên thì **[đọc trên màn hình]**. Trên toàn bộ test, MRR gián tiếp tăng từ 0.0475 lên 0.3143."
- **Thao tác (ca thất bại, khoảng 15 giây):** chọn `[thất bại, không cứu được] … Arthur C. Clarke by influenced. (CLIP 19 → 366)`.
- **Lời:** "Ở ca thất bại, mô hình trả về chính thực thể neo. Nút neo cũng là ảnh trả về, và đồ thị chỉ ra bộ ba đã dẫn mô hình đến
  kết quả sai."

### Đoạn 3 · 2:35 đến 3:55 · Cấp 3, ViHeritage (tiếng Việt)
- **Thao tác:**
  1. Sang tab cấp 3, chọn mô hình **CLIP + GAT**, cách diễn đạt **Câu hỏi**.
  2. Chọn ca `[thành công] Thành phố Huế có những di tích nào?`.
  3. Đổi cách diễn đạt sang **Câu mô tả**, rồi đổi mô hình sang **CLIP + AdaMF**.
- **Màn hình:**
  - Neo: Huế · Quan hệ: nằm ở tỉnh; ảnh hạng 1 là Chùa Thiên Mụ ✓.
  - Đồ thị: cạnh dày nhất "thuộc quần thể" tới Quần thể di tích Cố đô Huế; nét chấm nối quần thể này với Huế; thêm các di tích
    khác nằm ở tỉnh Huế.
  - Với câu mô tả "ảnh địa danh ở xứ Huế bên dòng sông Hương", ảnh hạng 1 **[kiểm tra trên màn hình]**.
  - Với AdaMF, giải thích chỉ còn một bộ ba.
- **Lời:** "Cấp 3 chạy trên ViHeritage-MMKG, đồ thị tri thức tiếng Việt về di sản do nhóm tự xây: 150 thực thể, 427 bộ ba, 438 ảnh
  từ Wikidata và Wikimedia Commons, có data card ghi nguồn và giấy phép. Câu hỏi về di tích ở Huế trả về Chùa Thiên Mụ ở hạng 1.
  Đồ thị cho thấy đường đi hai bước: Chùa Thiên Mụ thuộc Quần thể di tích Cố đô Huế, quần thể này nằm ở tỉnh Huế. Đổi sang câu mô
  tả, kết quả **[đọc]**: với truy vấn gián tiếp, điểm chủ yếu đến từ neo và quan hệ, không phụ thuộc cách đặt câu. AdaMF cũng trả
  về Chùa Thiên Mụ, và giải thích của nó chỉ còn đúng bộ ba này."
- **Thao tác (ca thất bại, khoảng 15 giây):** vẫn AdaMF, chọn `[thất bại] ảnh một ngôi chùa Phật giáo`. Lăng Minh Mạng hạng 1 ✗.
- **Lời:** "Câu hỏi tìm chùa nhưng mô hình trả về Lăng Minh Mạng. Đồ thị chỉ ra bộ ba 'Lăng Minh Mạng nằm ở tỉnh Huế', không liên quan
  quan hệ 'là một' được hỏi. Giải thích trung thành với mô hình, nhưng không bảo đảm hợp lý theo kiến thức miền."

### Đoạn 4 · 3:55 đến 4:35 · Tổng kết (một slide)

Ảnh slide có sẵn: `demo/slide_tong_ket.png` (nguồn `demo/slide_tong_ket.html`). Nội dung:

| Cấp | Dữ liệu (MRR test) | CLIP | CLIP + đồ thị | Ablation | Fid+ (ngẫu nhiên) |
|---|---|---|---|---|---|
| 1 | Visual Genome | 0.0611 | 0.0718 | bỏ cạnh vẫn giữ 77% mức tăng | 0.118 (0.003) |
| 2 | DB15K, gián tiếp | 0.0475 | 0.3143 | xáo cạnh chỉ giữ 37% | 0.511 (0.023) |
| 2 | FB15K-237-IMG, gián tiếp | 0.0122 | 0.1186 | xáo cạnh vẫn giữ 93% | 0.335 (0.011) |
| 3 | ViHeritage, gián tiếp | 0.112 | 0.219 AdaMF, 0.258 GAT | bỏ cạnh mất toàn bộ | 0.535 (0.145) AdaMF, 0.103 (0.002) GAT |

- **Lời:** "Đồ thị giúp rõ nhất ở truy vấn gián tiếp. Trên DB15K và ViHeritage, lợi ích cần đúng cạnh thật; trên Visual Genome và
  FB15K-237-IMG chưa chứng minh được vai trò của cấu trúc. Ở mọi tập, bỏ tập giải thích làm điểm giảm nhiều hơn hẳn bỏ ngẫu nhiên."

### Đoạn 5 · 4:35 đến 4:45 · Kết
- **Lời:** "Mã nguồn, tập chia dữ liệu, data card và báo cáo đã nộp kèm. Cảm ơn thầy cô."

## 3. Câu hỏi có thể gặp khi vấn đáp

1. **"Đề là tìm ảnh, sao truy vấn lại là câu hỏi?"**
   - Nhánh A chỉ yêu cầu đầu vào là văn bản, đầu ra là ảnh. Câu hỏi "Thành phố Huế có những di tích nào?" là truy vấn gián tiếp
     đúng kiểu đề nêu (child, plays, football): ảnh Chùa Thiên Mụ không cho thấy "ở Huế", phải đi qua quan hệ *nằm ở tỉnh*.
   - Mỗi truy vấn ViHeritage có ba cách nói: câu hỏi, câu mô tả và tiếng Anh. Số tiếng Việt trong báo cáo lấy trung bình hai cách nói;
     với câu tiếng Anh dạng mô tả, đồ thị vẫn hơn CLIP (0.248 và 0.266 so với 0.138).
   - DB15K, FB15K-237-IMG dùng câu dạng "Find an image of the entity …".
2. **"Truy vấn trực tiếp DB15K sao không có đồ thị?"** Theo thiết kế, loại truy vấn này chỉ xếp hạng bằng CLIP. Vẽ bộ ba khi đó là
   giải thích hậu kiểm, đề cấm.
3. **"Giải thích có phải vẽ thêm sau khi có kết quả?"** Không. Δ là điểm giảm khi bỏ phần tử đó và chấm lại bằng chính mô hình. Bỏ S
   thì hạng đổi, bỏ ngẫu nhiên cùng số thì gần như không (Fid+ ở bảng tổng kết).
4. **"Vì sao α_CLIP gần 0 ở truy vấn gián tiếp?"** α chọn trên validation. CLIP không thấy quan hệ trong ảnh nên validation chọn dựa vào
   đồ thị; CLIP vẫn tạo tập ứng viên và xếp hạng truy vấn trực tiếp.
5. **"Có rò rỉ dữ liệu không?"** Bộ ba giải thích chỉ là bộ ba train, không trùng validation/test; α, K* chọn trên validation; ảnh
   ứng viên là ảnh test.
6. **"GAT hay AdaMF tốt hơn trên ViHeritage?"** Không phân biệt được trên test (hiệu 0.039, p = 0.29).

## 4. Lưu ý theo từng cấp

| | Cấp 1: Visual Genome | Cấp 2: DB15K | Cấp 2: FB15K-237-IMG | Cấp 3: ViHeritage |
|---|---|---|---|---|
| Truy vấn | câu tiếng Anh tự do, không có neo | gián tiếp: neo + quan hệ; trực tiếp: câu tự do | như DB15K | câu tiếng Việt + neo + quan hệ |
| Ứng viên | CLIP top-50 trong 10 443 ảnh test | CLIP top-1000 | CLIP top-1000 | toàn bộ ảnh test |
| α_CLIP | 0.6 | 0.0 hoặc 0.1 | 0.1 | AdaMF 0.0; GAT 0.0 hoặc 0.1 |
| Phần tử giải thích | quan hệ scene graph, cạnh "ảnh chứa vật" | bộ ba train | bộ ba train | bộ ba train |
| Cách đo Δ | bỏ quan hệ, chấm lại GAT (Δ cỡ 10⁻³) | học lại nhúng của thực thể | như DB15K | AdaMF: học lại nhúng; GAT: che cạnh |
| K* | 12 | 2 | 8 | AdaMF 1; GAT 8 |
| Không được nói | "cải thiện nhờ quan hệ" (bỏ cạnh vẫn giữ 77%) | | "cải thiện nhờ cấu trúc" (xáo cạnh giữ 93%) | "GAT tốt hơn AdaMF" |

**Cấp 1 (Visual Genome):**
- Khoảng 75% phần tử giải thích là cạnh "ảnh chứa vật". Hãy nói mô hình dựa vào vật có trong ảnh.
- Attention không dùng làm giải thích (Fid+ −0.007).

**Cấp 2:**
- Cạnh chấm xanh nhạt chỉ để nối về neo, không phải phần tử được đo.
- Ảnh DB15K là ảnh đại diện của thực thể nên có thể trông không liên quan (ví dụ bản đồ). Dữ liệu có bộ ba nhiễu như
  "Maurice Sendak type Island".

**Cấp 3:**
- Với α_CLIP = 0, chữ trong câu không đổi thứ hạng của truy vấn gián tiếp; đây là lý do ca Lăng Minh Mạng.
- Hình Huế trong báo cáo dùng GAT (8 bộ ba). Nói rõ đang chọn mô hình nào.
- AdaMF ở K* = 1: Fid+ cao nhưng PS 0.144, tức một bộ ba cần thiết nhưng chưa đủ.

**Chung:**
- Nói "ba lần huấn luyện", không nói seed.
- Giải thích trung thành với mô hình không có nghĩa là đúng theo kiến thức.
