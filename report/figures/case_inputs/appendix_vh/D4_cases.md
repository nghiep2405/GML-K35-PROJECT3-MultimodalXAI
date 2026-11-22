# Ca phân tích XAI trên ViHeritage-MMKG (K* = {'adamf': 1, 'gat': 8}; seed của từng ca ghi trong tiêu đề)

## adamf_01 — AdaMF, thành công (seed 2027)
- Câu: “Thành phố Huế có những di tích nào?” (anchor **Huế**, quan hệ *nằm ở tỉnh*, chiều head)
- Đáp án: Chùa Thiên Mụ; ảnh hạng 1: **Chùa Thiên Mụ**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 1):
  - Chùa Thiên Mụ –[thuộc quần thể]→ Quần thể di tích Cố đô Huế  (δ = +0.227; đường đi: Quần thể di tích Cố đô Huế –[nằm ở tỉnh]→ Huế ⇒ Chùa Thiên Mụ –[thuộc quần thể]→ Quần thể di tích Cố đô Huế)
- Fid+ 0.268 (ngẫu nhiên 0.062), Fid− 0.983, PN 0, PS 0, làm nhiễu -0.491, sparsity 0.86

## adamf_02 — AdaMF, thành công (seed 2027)
- Câu: “Gần Tháp Bà Ponagar ở Nha Trang có công trình nào khác?” (anchor **Tháp Po Nagar**, quan hệ *ở gần*, chiều head)
- Đáp án: Nhà thờ Núi; ảnh hạng 1: **Nhà thờ Núi**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 1):
  - Nhà thờ Núi –[ở gần]→ Chùa Long Sơn  (δ = +0.388; đường đi: Chùa Long Sơn –[ở gần]→ Tháp Po Nagar ⇒ Nhà thờ Núi –[ở gần]→ Chùa Long Sơn)
- Fid+ 0.379 (ngẫu nhiên 0.037), Fid− 0.405, PN 0, PS 1, làm nhiễu -0.160, sparsity 0.83

## adamf_03 — AdaMF, thành công (seed 2027)
- Câu: “Hoàng thành Huế là một phần của quần thể di sản nào?” (anchor **Hoàng thành Huế**, quan hệ *thuộc quần thể*, chiều tail)
- Đáp án: Quần thể di tích Cố đô Huế; ảnh hạng 1: **Quần thể di tích Cố đô Huế**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 1):
  - Lăng Minh Mạng –[thuộc quần thể]→ Quần thể di tích Cố đô Huế  (δ = +0.301; đường đi: Lăng Minh Mạng –[thuộc quần thể]→ Hoàng thành Huế ⇒ Lăng Minh Mạng –[thuộc quần thể]→ Quần thể di tích Cố đô Huế)
- Fid+ 0.308 (ngẫu nhiên 0.028), Fid− 0.205, PN 1, PS 1, làm nhiễu -0.128, sparsity 0.86

## adamf_04 — AdaMF, thành công (seed 2027)
- Câu: “Cho mình xem các địa danh ở Hà Nội” (anchor **Hà Nội**, quan hệ *nằm ở tỉnh*, chiều head)
- Đáp án: Tháp Rùa; ảnh hạng 1: **Tháp Rùa**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 1):
  - Tháp Rùa –[ở gần]→ Hoàn Kiếm hồ  (δ = +0.136; đường đi: Hoàn Kiếm hồ –[nằm ở tỉnh]→ Hà Nội ⇒ Tháp Rùa –[ở gần]→ Hoàn Kiếm hồ)
- Fid+ 0.198 (ngẫu nhiên -0.159), Fid− 0.990, PN 0, PS 0, làm nhiễu 0.322, sparsity 0.89

## adamf_05 — AdaMF, thất bại (seed 2027)
- Câu: “ảnh một ngôi chùa Phật giáo” (anchor **chùa**, quan hệ *là một*, chiều head)
- Đáp án: Chùa Xá Lợi; ảnh hạng 1: **Lăng Minh Mạng**; ảnh đúng tốt nhất ở hạng 17
- Giải thích (K* = 1):
  - Lăng Minh Mạng –[nằm ở tỉnh]→ Huế  (δ = +1.029; ngoài đường đi anchor–e*)
- Fid+ 0.842 (ngẫu nhiên -0.286), Fid− 0.939, PN 1, PS 0, làm nhiễu 0.548, sparsity 0.86

## adamf_06 — AdaMF, thất bại (seed 2027)
- Câu: “Quanh Lăng Bác ở quảng trường Ba Đình có những công trình nào?” (anchor **Lăng Chủ tịch Hồ Chí Minh**, quan hệ *ở gần*, chiều head)
- Đáp án: Phủ Chủ tịch; ảnh hạng 1: **Lăng Chủ tịch Hồ Chí Minh**; ảnh đúng tốt nhất ở hạng 8
- Giải thích (K* = 1):
  - Lăng Chủ tịch Hồ Chí Minh –[là một]→ lăng  (δ = +0.759; đường đi: Lăng Chủ tịch Hồ Chí Minh –[là một]→ lăng ⇒ Lăng Chủ tịch Hồ Chí Minh –[là một]→ lăng)
- Fid+ 0.598 (ngẫu nhiên 0.363), Fid− 0.921, PN 1, PS 0, làm nhiễu 0.554, sparsity 0.67

## adamf_07 — AdaMF, thất bại (seed 2027)
- Câu: “Quanh chùa Giác Lâm ở Tân Bình có nơi nào đáng xem?” (anchor **Chùa Giác Lâm**, quan hệ *ở gần*, chiều head)
- Đáp án: Chùa Xá Lợi; ảnh hạng 1: **Chùa Giác Lâm**; ảnh đúng tốt nhất ở hạng 21
- Giải thích (K* = 1):
  - Chùa Giác Lâm –[ở gần]→ Chùa Bà Thiên Hậu  (δ = +0.573; đường đi: Chùa Bà Thiên Hậu –[ở gần]→ Chùa Giác Lâm ⇒ Chùa Giác Lâm –[ở gần]→ Chùa Bà Thiên Hậu)
- Fid+ 0.440 (ngẫu nhiên 0.092), Fid− -0.006, PN 0, PS 1, làm nhiễu -0.220, sparsity 0.75

## adamf_08 — AdaMF, thất bại (seed 2027)
- Câu: “Quanh Lăng Bác ở quảng trường Ba Đình có những công trình nào?” (anchor **Lăng Chủ tịch Hồ Chí Minh**, quan hệ *ở gần*, chiều head)
- Đáp án: Chùa Một Cột; ảnh hạng 1: **Lăng Chủ tịch Hồ Chí Minh**; ảnh đúng tốt nhất ở hạng 6
- Giải thích (K* = 1):
  - Lăng Chủ tịch Hồ Chí Minh –[xây dựng vào]→ thế kỷ 20  (δ = +0.905; đường đi: Lăng Chủ tịch Hồ Chí Minh –[xây dựng vào]→ thế kỷ 20 ⇒ Lăng Chủ tịch Hồ Chí Minh –[xây dựng vào]→ thế kỷ 20)
- Fid+ 0.667 (ngẫu nhiên 0.359), Fid− 0.801, PN 1, PS 0, làm nhiễu 0.351, sparsity 0.67

## adamf_09 — AdaMF, thất bại (seed 2027)
- Câu: “Những nhà thờ nào trực thuộc Tổng giáo phận Hà Nội?” (anchor **Tổng giáo phận Hà Nội**, quan hệ *thuộc giáo phận*, chiều head)
- Đáp án: Vương cung thánh đường Sở Kiện; ảnh hạng 1: **Bảo tàng Lịch sử Việt Nam**; ảnh đúng tốt nhất ở hạng 6
- Giải thích (K* = 1):
  - Bảo tàng Lịch sử Việt Nam –[kiến trúc sư]→ Ernest Hébrard  (δ = +0.734; ngoài đường đi anchor–e*)
- Fid+ 0.569 (ngẫu nhiên 0.165), Fid− 0.739, PN 1, PS 0, làm nhiễu 0.600, sparsity 0.80

## adamf_10 — AdaMF, thành công (seed 2028)
- Câu: “Công trình nào là một phần của khu Di sản thế giới UNESCO?” (anchor **một phần của di sản thế giới UNESCO**, quan hệ *danh hiệu di sản*, chiều head)
- Đáp án: Lăng Minh Mạng; ảnh hạng 1: **Lăng Minh Mạng**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 1):
  - Lăng Minh Mạng –[nằm ở tỉnh]→ Huế  (δ = +0.731; ngoài đường đi anchor–e*)
- Fid+ 0.641 (ngẫu nhiên 0.052), Fid− 0.860, PN 1, PS 0, làm nhiễu 0.648, sparsity 0.86

## gat_01 — GAT, thành công (seed 2027)
- Câu: “Những nhà thờ nào trực thuộc Tổng giáo phận Hà Nội?” (anchor **Tổng giáo phận Hà Nội**, quan hệ *thuộc giáo phận*, chiều head)
- Đáp án: Vương cung thánh đường Sở Kiện; ảnh hạng 1: **Vương cung thánh đường Sở Kiện**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 8):
  - Vương cung thánh đường Sở Kiện –[là một]→ nhà thờ Kitô giáo  (δ = +0.084; ngoài đường đi anchor–e*)
  - Nhà thờ Hàm Long –[thuộc giáo phận]→ Tổng giáo phận Hà Nội  (δ = +0.056; ngoài đường đi anchor–e*)
  - Nhà thờ Hàm Long –[xây dựng vào]→ thế kỷ 20  (δ = +0.009; ngoài đường đi anchor–e*)
  - Nhà thờ Cửa Bắc –[là một]→ nhà thờ Kitô giáo  (δ = +0.009; ngoài đường đi anchor–e*)
  - Nhà thờ Cửa Bắc –[kiến trúc sư]→ Ernest Hébrard  (δ = +0.007; ngoài đường đi anchor–e*)
  - Nhà thờ Hàm Long –[nằm ở tỉnh]→ Hà Nội  (δ = +0.006; ngoài đường đi anchor–e*)
  - Nhà thờ Hàm Long –[là một]→ nhà thờ Kitô giáo  (δ = +0.002; ngoài đường đi anchor–e*)
  - Nhà thờ chính tòa Phát Diệm –[xây dựng vào]→ thế kỷ 19  (δ = +0.002; ngoài đường đi anchor–e*)
- Fid+ 0.069 (ngẫu nhiên 0.025), Fid− -0.094, PN 0, PS 1, làm nhiễu 0.025, sparsity 0.73

## gat_02 — GAT, thành công (seed 2027)
- Câu: “Thành phố Huế có những di tích nào?” (anchor **Huế**, quan hệ *nằm ở tỉnh*, chiều head)
- Đáp án: Chùa Thiên Mụ; ảnh hạng 1: **Chùa Thiên Mụ**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 8):
  - Chùa Thiên Mụ –[thuộc quần thể]→ Quần thể di tích Cố đô Huế  (δ = +0.082; đường đi: Quần thể di tích Cố đô Huế –[nằm ở tỉnh]→ Huế ⇒ Chùa Thiên Mụ –[thuộc quần thể]→ Quần thể di tích Cố đô Huế)
  - Hoàng thành Huế –[nằm ở tỉnh]→ Huế  (δ = +0.081; đường đi: Hoàng thành Huế –[nằm ở tỉnh]→ Huế ⇒ Chùa Thiên Mụ –[thuộc quần thể]→ Hoàng thành Huế)
  - Nhà thờ chính tòa Phủ Cam –[nằm ở tỉnh]→ Huế  (δ = +0.061; ngoài đường đi anchor–e*)
  - Lăng Tự Đức –[nằm ở tỉnh]→ Huế  (δ = +0.038; đường đi: Lăng Tự Đức –[nằm ở tỉnh]→ Huế ⇒ Lăng Tự Đức –[ở gần]→ Chùa Thiên Mụ)
  - Lăng Khải Định –[nằm ở tỉnh]→ Huế  (δ = +0.032; ngoài đường đi anchor–e*)
  - Lăng Minh Mạng –[nằm ở tỉnh]→ Huế  (δ = +0.030; ngoài đường đi anchor–e*)
  - Chùa Thiên Mụ –[ở gần]→ Lăng Tự Đức  (δ = +0.015; đường đi: Lăng Tự Đức –[nằm ở tỉnh]→ Huế ⇒ Chùa Thiên Mụ –[ở gần]→ Lăng Tự Đức)
  - Lăng Tự Đức –[ở gần]→ Chùa Thiên Mụ  (δ = +0.014; đường đi: Lăng Tự Đức –[nằm ở tỉnh]→ Huế ⇒ Lăng Tự Đức –[ở gần]→ Chùa Thiên Mụ)
- Fid+ 0.487 (ngẫu nhiên 0.008), Fid− -0.311, PN 0, PS 1, làm nhiễu 0.155, sparsity 0.83

## gat_03 — GAT, thành công (seed 2027)
- Câu: “Quần thể di tích Cố đô Huế gồm những công trình nào?” (anchor **Quần thể di tích Cố đô Huế**, quan hệ *thuộc quần thể*, chiều head)
- Đáp án: Lăng Tự Đức; ảnh hạng 1: **Lăng Tự Đức**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 8):
  - Chùa Thiên Mụ –[thuộc quần thể]→ Quần thể di tích Cố đô Huế  (δ = +0.134; đường đi: Chùa Thiên Mụ –[thuộc quần thể]→ Quần thể di tích Cố đô Huế ⇒ Lăng Tự Đức –[ở gần]→ Chùa Thiên Mụ)
  - Lăng Minh Mạng –[thuộc quần thể]→ Quần thể di tích Cố đô Huế  (δ = +0.110; đường đi: Lăng Minh Mạng –[thuộc quần thể]→ Quần thể di tích Cố đô Huế ⇒ Lăng Minh Mạng –[ở gần]→ Lăng Tự Đức)
  - Lăng Tự Đức –[nằm ở tỉnh]→ Huế  (δ = +0.024; đường đi: Quần thể di tích Cố đô Huế –[nằm ở tỉnh]→ Huế ⇒ Lăng Tự Đức –[nằm ở tỉnh]→ Huế)
  - Quần thể di tích Cố đô Huế –[nằm ở tỉnh]→ Huế  (δ = +0.019; đường đi: Quần thể di tích Cố đô Huế –[nằm ở tỉnh]→ Huế ⇒ Lăng Tự Đức –[nằm ở tỉnh]→ Huế)
  - lăng –[loại con của]→ tượng đài  (δ = +0.005; ngoài đường đi anchor–e*)
  - Nhà thờ chính tòa Phủ Cam –[tôn giáo]→ Công giáo  (δ = +0.005; ngoài đường đi anchor–e*)
  - Lăng Tự Đức –[là một]→ lăng  (δ = +0.004; ngoài đường đi anchor–e*)
  - Lăng Khải Định –[xây dựng vào]→ thế kỷ 20  (δ = +0.003; ngoài đường đi anchor–e*)
- Fid+ 0.194 (ngẫu nhiên 0.008), Fid− -0.051, PN 1, PS 0, làm nhiễu 0.074, sparsity 0.83

## gat_04 — GAT, thành công (seed 2027)
- Câu: “Công trình nào ở Việt Nam mang phong cách Tân Gothic?” (anchor **kiến trúc Tân Gothic**, quan hệ *phong cách kiến trúc*, chiều head)
- Đáp án: Nhà thờ Núi; ảnh hạng 1: **Nhà thờ Núi**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 8):
  - Nhà thờ Mằng Lăng –[phong cách kiến trúc]→ kiến trúc Tân Gothic  (δ = +0.050; ngoài đường đi anchor–e*)
  - Nhà thờ Núi –[xây dựng vào]→ thế kỷ 20  (δ = +0.015; ngoài đường đi anchor–e*)
  - Nhà thờ Núi –[ở gần]→ Chùa Long Sơn  (δ = +0.013; ngoài đường đi anchor–e*)
  - Chùa Long Sơn –[ở gần]→ Nhà thờ Núi  (δ = +0.010; ngoài đường đi anchor–e*)
  - Nhà thờ Mằng Lăng –[là một]→ nhà thờ Kitô giáo  (δ = +0.007; ngoài đường đi anchor–e*)
  - Nhà tù Hỏa Lò –[ở gần]→ Nhà thờ Lớn Hà Nội  (δ = +0.003; ngoài đường đi anchor–e*)
  - Nhà thờ Lớn Hà Nội –[ở gần]→ Nhà tù Hỏa Lò  (δ = +0.003; ngoài đường đi anchor–e*)
  - Chùa Long Sơn –[tôn giáo]→ Phật giáo  (δ = +0.003; ngoài đường đi anchor–e*)
- Fid+ 0.085 (ngẫu nhiên 0.007), Fid− -0.042, PN 0, PS 1, làm nhiễu 0.021, sparsity 0.83

## gat_05 — GAT, thành công (seed 2027)
- Câu: “Đồng Nai có những địa danh nào đáng đến?” (anchor **Đồng Nai**, quan hệ *nằm ở tỉnh*, chiều head)
- Đáp án: Vườn quốc gia Cát Tiên; ảnh hạng 1: **Vườn quốc gia Cát Tiên**; ảnh đúng tốt nhất ở hạng 1
- Giải thích (K* = 8):
  - Vườn quốc gia Cát Tiên –[nằm ở tỉnh]→ Lâm Đồng  (δ = +0.278; đường đi: Lâm Đồng –[giáp ranh]→ Đồng Nai ⇒ Vườn quốc gia Cát Tiên –[nằm ở tỉnh]→ Lâm Đồng)
  - Lâm Đồng –[giáp ranh]→ Đồng Nai  (δ = +0.003; đường đi: Lâm Đồng –[giáp ranh]→ Đồng Nai ⇒ Vườn quốc gia Cát Tiên –[nằm ở tỉnh]→ Lâm Đồng)
  - Đồng Nai –[giáp ranh]→ Lâm Đồng  (δ = +0.003; đường đi: Đồng Nai –[giáp ranh]→ Lâm Đồng ⇒ Vườn quốc gia Cát Tiên –[nằm ở tỉnh]→ Lâm Đồng)
  - Khánh Hòa –[giáp ranh]→ Lâm Đồng  (δ = +0.000; ngoài đường đi anchor–e*)
  - Lâm Đồng –[giáp ranh]→ Khánh Hòa  (δ = +0.000; ngoài đường đi anchor–e*)
- Fid+ 0.498 (ngẫu nhiên 0.096), Fid− -0.152, PN 1, PS 1, làm nhiễu -0.084, sparsity 0.50

## gat_06 — GAT, thất bại (seed 2027)
- Câu: “ảnh một ngôi chùa Phật giáo” (anchor **chùa**, quan hệ *là một*, chiều head)
- Đáp án: Chùa Xá Lợi; ảnh hạng 1: **Đền Quán Thánh**; ảnh đúng tốt nhất ở hạng 29
- Giải thích (K* = 8):
  - Đền Quán Thánh –[nằm ở tỉnh]→ Hà Nội  (δ = +0.069; ngoài đường đi anchor–e*)
  - Chùa Trấn Quốc –[là một]→ chùa  (δ = +0.053; đường đi: Chùa Trấn Quốc –[là một]→ chùa ⇒ Đền Quán Thánh –[ở gần]→ Chùa Trấn Quốc)
  - Chùa Giác Lâm –[là một]→ chùa  (δ = +0.019; ngoài đường đi anchor–e*)
  - Chùa Một Cột –[là một]→ chùa  (δ = +0.018; ngoài đường đi anchor–e*)
  - Chùa Một Cột –[ở gần]→ Phủ Chủ tịch  (δ = +0.005; ngoài đường đi anchor–e*)
  - Phủ Chủ tịch –[ở gần]→ Chùa Một Cột  (δ = +0.005; ngoài đường đi anchor–e*)
  - Chùa Trấn Quốc –[tôn giáo]→ Phật giáo  (δ = +0.004; ngoài đường đi anchor–e*)
  - Chùa Trấn Quốc –[nằm ở tỉnh]→ Hà Nội  (δ = +0.004; ngoài đường đi anchor–e*)
- Fid+ 0.141 (ngẫu nhiên -0.014), Fid− -0.250, PN 0, PS 1, làm nhiễu 0.016, sparsity 0.87

## gat_07 — GAT, thất bại (seed 2027)
- Câu: “Quanh Lăng Bác ở quảng trường Ba Đình có những công trình nào?” (anchor **Lăng Chủ tịch Hồ Chí Minh**, quan hệ *ở gần*, chiều head)
- Đáp án: Phủ Chủ tịch; ảnh hạng 1: **Lăng Chủ tịch Hồ Chí Minh**; ảnh đúng tốt nhất ở hạng 25
- Giải thích (K* = 8):
  - Lăng Chủ tịch Hồ Chí Minh –[nằm ở tỉnh]→ Hà Nội  (δ = +0.065; đường đi: Lăng Chủ tịch Hồ Chí Minh –[nằm ở tỉnh]→ Hà Nội ⇒ Lăng Chủ tịch Hồ Chí Minh –[nằm ở tỉnh]→ Hà Nội)
  - lăng –[loại con của]→ tượng đài  (δ = +0.006; ngoài đường đi anchor–e*)
  - Nhà thờ Núi –[xây dựng vào]→ thế kỷ 20  (δ = +0.003; ngoài đường đi anchor–e*)
  - Hà Nội –[giáp ranh]→ Bắc Ninh  (δ = +0.001; ngoài đường đi anchor–e*)
  - Nhà thờ Hàm Long –[xây dựng vào]→ thế kỷ 20  (δ = +0.001; ngoài đường đi anchor–e*)
  - Bắc Ninh –[giáp ranh]→ Hà Nội  (δ = +0.001; ngoài đường đi anchor–e*)
  - Hà Nội –[giáp ranh]→ Thái Nguyên  (δ = +0.001; ngoài đường đi anchor–e*)
  - Thái Nguyên –[giáp ranh]→ Hà Nội  (δ = +0.001; ngoài đường đi anchor–e*)
- Fid+ -0.008 (ngẫu nhiên 0.001), Fid− -0.129, PN 0, PS 1, làm nhiễu -0.061, sparsity 0.76

## gat_08 — GAT, thất bại (seed 2027)
- Câu: “Quanh chùa Giác Lâm ở Tân Bình có nơi nào đáng xem?” (anchor **Chùa Giác Lâm**, quan hệ *ở gần*, chiều head)
- Đáp án: Chùa Xá Lợi; ảnh hạng 1: **Chùa Giác Lâm**; ảnh đúng tốt nhất ở hạng 4
- Giải thích (K* = 8):
  - Chùa Giác Lâm –[ở gần]→ Chùa Bà Thiên Hậu  (δ = +0.024; đường đi: Chùa Bà Thiên Hậu –[ở gần]→ Chùa Giác Lâm ⇒ Chùa Giác Lâm –[ở gần]→ Chùa Bà Thiên Hậu)
  - Chùa Bà Thiên Hậu –[ở gần]→ Chùa Giác Lâm  (δ = +0.022; đường đi: Chùa Bà Thiên Hậu –[ở gần]→ Chùa Giác Lâm ⇒ Chùa Bà Thiên Hậu –[ở gần]→ Chùa Giác Lâm)
  - Chùa Thiên Mụ –[là một]→ chùa  (δ = +0.001; ngoài đường đi anchor–e*)
  - Nhà thờ Đức Bà Sài Gòn –[nằm ở tỉnh]→ Thành phố Hồ Chí Minh  (δ = +0.001; ngoài đường đi anchor–e*)
  - Bảo tàng Lịch sử Thành phố Hồ Chí Minh –[nằm ở tỉnh]→ Thành phố Hồ Chí Minh  (δ = +0.001; ngoài đường đi anchor–e*)
  - Chùa Bái Đính –[là một]→ chùa  (δ = +0.001; ngoài đường đi anchor–e*)
  - Chùa Bút Tháp –[là một]→ chùa  (δ = +0.000; ngoài đường đi anchor–e*)
  - Nhà hát Sài Gòn –[nằm ở tỉnh]→ Thành phố Hồ Chí Minh  (δ = +0.000; ngoài đường đi anchor–e*)
- Fid+ -0.007 (ngẫu nhiên -0.003), Fid− -0.018, PN 0, PS 1, làm nhiễu -0.022, sparsity 0.60

## gat_09 — GAT, thất bại (seed 2027)
- Câu: “Quanh Lăng Bác ở quảng trường Ba Đình có những công trình nào?” (anchor **Lăng Chủ tịch Hồ Chí Minh**, quan hệ *ở gần*, chiều head)
- Đáp án: Chùa Một Cột; ảnh hạng 1: **Lăng Chủ tịch Hồ Chí Minh**; ảnh đúng tốt nhất ở hạng 8
- Giải thích (K* = 8):
  - Lăng Chủ tịch Hồ Chí Minh –[nằm ở tỉnh]→ Hà Nội  (δ = +0.065; đường đi: Lăng Chủ tịch Hồ Chí Minh –[nằm ở tỉnh]→ Hà Nội ⇒ Lăng Chủ tịch Hồ Chí Minh –[nằm ở tỉnh]→ Hà Nội)
  - lăng –[loại con của]→ tượng đài  (δ = +0.006; ngoài đường đi anchor–e*)
  - Nhà thờ Núi –[xây dựng vào]→ thế kỷ 20  (δ = +0.003; ngoài đường đi anchor–e*)
  - Hà Nội –[giáp ranh]→ Bắc Ninh  (δ = +0.001; ngoài đường đi anchor–e*)
  - Nhà thờ Hàm Long –[xây dựng vào]→ thế kỷ 20  (δ = +0.001; ngoài đường đi anchor–e*)
  - Bắc Ninh –[giáp ranh]→ Hà Nội  (δ = +0.001; ngoài đường đi anchor–e*)
  - Hà Nội –[giáp ranh]→ Thái Nguyên  (δ = +0.001; ngoài đường đi anchor–e*)
  - Thái Nguyên –[giáp ranh]→ Hà Nội  (δ = +0.001; ngoài đường đi anchor–e*)
- Fid+ -0.007 (ngẫu nhiên -0.000), Fid− -0.131, PN 0, PS 1, làm nhiễu -0.030, sparsity 0.76

## gat_10 — GAT, thất bại (seed 2027)
- Câu: “Những nơi nào của Việt Nam đang nằm trong danh sách dự kiến đề cử Di sản thế giới?” (anchor **danh sách dự kiến di sản thế giới**, quan hệ *danh hiệu di sản*, chiều head)
- Đáp án: Óc Eo; ảnh hạng 1: **Vịnh Hạ Long**; ảnh đúng tốt nhất ở hạng 15
- Giải thích (K* = 8):
  - Vịnh Hạ Long –[danh hiệu di sản]→ di sản thế giới  (δ = +1.651; ngoài đường đi anchor–e*)
  - Cát Bà –[danh hiệu di sản]→ danh sách dự kiến di sản thế giới  (δ = +0.097; ngoài đường đi anchor–e*)
  - Cát Bà –[danh hiệu di sản]→ di sản thế giới  (δ = +0.067; ngoài đường đi anchor–e*)
  - Vườn quốc gia Ba Bể –[nằm ở tỉnh]→ Thái Nguyên  (δ = +0.021; ngoài đường đi anchor–e*)
  - Quần thể danh thắng Tràng An –[danh hiệu di sản]→ di sản thế giới  (δ = +0.002; ngoài đường đi anchor–e*)
- Fid+ 0.648 (ngẫu nhiên 0.343), Fid− -0.173, PN 1, PS 1, làm nhiễu -0.376, sparsity 0.62
