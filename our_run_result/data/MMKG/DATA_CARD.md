# ViHeritage-MMKG — Data card

MMKG tiếng Việt nhỏ về di sản – danh thắng Việt Nam, tự xây cho Project 03 (Học máy với dữ liệu đồ thị, cấp độ 3).
Ngày chụp dữ liệu: **2026-11-15**. Mọi con số dưới đây do notebook `08b` sinh từ file.

## Quy mô
| | Số lượng |
|---|---|
| Thực thể | 150 (88 có ảnh) |
| Loại thực thể | structure 73, concept 32, province 21, natural 15, period 3, person 3, diocese 1, feature 1, site 1 |
| Quan hệ | 15 |
| Triple | 427 (train 318, valid 53, test 56); trong đó 151 triple suy ra |
| Ảnh | 438 (train 264, val 86, test 88) |
| Truy vấn | test/direct: 88, test/indirect: 54, val/direct: 86, val/indirect: 50 |

## Nguồn
- **Thực thể, quan hệ**: Wikidata (statement truthy, hạng tốt nhất), API `wbgetentities` và SPARQL `query.wikidata.org`. CC0.
- **Văn bản**: nhãn, bí danh, mô tả Wikidata (CC0); 2–3 câu đầu phần mở bài Wikipedia tiếng Việt (CC BY-SA 4.0, ghi `wiki_revid_vi`).
- **Ảnh**: Wikimedia Commons, thumbnail 960 px; mỗi ảnh ghi tác giả, giấy phép, trang file (`LICENSES.md`).
- Nhãn tiếng Việt do nhóm dịch: 7 khái niệm chỉ có nhãn tiếng Anh trên Wikidata;
  1 thực thể vẫn dùng nhãn tiếng Anh (tên riêng).

## Chọn thực thể
Ứng viên: mục ở Việt Nam có danh hiệu di sản (P1435), hoặc thuộc loại công trình/danh thắng phổ biến (lọc theo nhãn loại, danh sách loại ở
`raw/class_whitelist.csv`), có nhãn tiếng Việt, bài Wikipedia tiếng Việt và ảnh. Xếp theo số sitelink. Tối đa 15 danh thắng thiên nhiên,
6 cầu, 8 bảo tàng. Thực thể không ảnh là đầu kia của quan hệ, xuất hiện ≥ 2 lần (tỉnh luôn giữ). Nhân vật chỉ gồm người đã mất.

## Quan hệ và cách tạo triple
| pid | label_vi | label_en | n_triples |
|---|---|---|---|
| P31 | là một | instance of | 86 |
| P131 | nằm ở tỉnh | located in province | 78 |
| P1435 | danh hiệu di sản | heritage designation | 18 |
| P140 | tôn giáo | religion | 19 |
| P361 | thuộc quần thể | part of | 11 |
| P825 | thờ / dành cho | dedicated to | 4 |
| P149 | phong cách kiến trúc | architectural style | 6 |
| P706 | nằm trên địa hình | located on terrain feature | 1 |
| P84 | kiến trúc sư | architect | 5 |
| P177 | bắc qua | crosses | 3 |
| P708 | thuộc giáo phận | diocese | 4 |
| NEARBY | ở gần | near (≤ 5 km) | 122 |
| CENTURY | xây dựng vào | built in century | 29 |
| P47 | giáp ranh | shares border with | 35 |
| P279 | loại con của | subclass of | 6 |

- `P131`: đi theo chuỗi P131 tới cấp tỉnh rồi đổi sang **34 tỉnh, thành sau sắp xếp từ 1/7/2025** (Nghị quyết 202/2025/QH15). Tỉnh bị
  gộp vẫn dùng tên tỉnh mới; ví dụ: An Giang (gộp từ: An Giang, Kiên Giang); Bắc Ninh (gộp từ: Bắc Ninh, Bắc Giang); Cần Thơ (gộp từ: Cần Thơ, Sóc Trăng, Hậu Giang); Hưng Yên (gộp từ: Hưng Yên, Thái Bình).
- `P47`: hai tỉnh mới giáp nhau khi hai tỉnh cũ thuộc chúng giáp nhau.
- `P361`: gồm cả `P527` (có bộ phận) đảo chiều.
- **Suy ra** (cột `source` bắt đầu bằng `derived`): `NEARBY` từ tọa độ P625 (mỗi thực thể có ảnh nối với 2 thực thể có ảnh gần nhất trong
  5 km, hai chiều); `CENTURY` từ năm khởi lập P571. Cả hai suy từ thuộc tính nằm ngoài KG, không dùng thống kê trên triple, nên không làm
  lộ triple test.

## Liên kết ảnh – thực thể
Ảnh lấy từ P18 (`P18`), category Commons P373 (`P373_d0`) và category con (`P373_d1`). Lọc: JPEG/PNG, cạnh ngắn ≥ 400 px, giấy phép tự do,
từ khóa tên file/category, CLIP zero-shot (bản đồ, tài liệu, chân dung, đồ ăn…), khử trùng SHA-256 và cosine CLIP ≥ 0,95.
Kiểm tra tay 100 ảnh ngẫu nhiên chia tầng (người chấm: A = Claude (AI, Anthropic) – chấm 17/11/2026 theo yêu cầu nhóm; không độc lập với bộ lọc ảnh; Cohen κ = không có):

| link_method | n | acc_A | khong_ro_A | acc_mean |
|---|---|---|---|---|
| P18 | 17 | 1.000 | 1 | 1.000 |
| P373_d0 | 76 | 0.929 | 6 | 0.929 |
| P373_d1 | 7 | 0.833 | 1 | 0.833 |
| tất cả | 100 | 0.935 | 8 | 0.935 |

Đã loại 12 ảnh (kể cả mọi ảnh P373_d1).

## Chia dữ liệu và chống rò rỉ
- Triple chia 70%/15%/15% theo từng quan hệ, seed 2025; cặp đối xứng chung split;
  thực thể của triple val/test có trong train. Mã: `source/MMKG/make_splits.py`.
- Ảnh chia theo thực thể; đặc trưng thị giác chỉ lấy từ ảnh train. Truy vấn val/test sinh từ triple và ảnh của chính split đó.
- Truy vấn gián tiếp đánh giá kiểu lọc: đáp án đúng khác của cùng (quan hệ, anchor) bị loại khỏi danh sách xếp hạng.
- Rò rỉ qua văn bản: tỉ lệ truy vấn gián tiếp có tên anchor trong mô tả/mở bài Wikipedia của đáp án = {'test': 0.389, 'val': 0.42}. Đây là thông tin hợp lệ lúc
  suy luận nhưng được báo cáo, và đo lại khi che văn bản (E2).

## Hạn chế
- Wikidata về di tích Việt Nam thưa (ít P1435, P825); khoảng 35% triple là quan hệ suy ra.
- Thiên về nơi nổi tiếng (xếp theo sitelink) và ảnh du lịch trên Commons; ít ảnh nội thất, ảnh cũ.
- Địa giới theo 2025; mô tả Wikipedia có thể còn dùng tên tỉnh cũ.

## Điều kiện sử dụng
Chỉ dùng cho học tập và nghiên cứu. Ảnh giữ giấy phép gốc (xem `LICENSES.md`, cần ghi tác giả khi tái sử dụng; ảnh CC BY-SA phải giữ
cùng giấy phép). Văn bản Wikipedia theo CC BY-SA 4.0; dữ liệu Wikidata CC0. Không chứa dữ liệu cá nhân nhạy cảm; nhân vật chỉ là nhân vật
lịch sử đã mất. Ảnh có thể có người qua đường; ảnh chân dung, nhóm người tạo dáng đã được lọc.
