# B9 — Phân tích chi tiết 5 truy vấn thành công và 5 truy vấn thất bại

Mô hình: `seed_2026`, α_graph = 0.2, K = 8 (chọn trên validation).


## success_1
**Truy vấn:** "A lady wearing a blue print shirt and green head cover smiles for the camera ."  
**Ảnh đúng:** 6160193920  
**Hạng:** CLIP 127 · CLIP+GAT 1 · no_edge 4 · random_edges 13  
**Điểm ảnh đúng:** CLIP 0.2226 · Graph 0.4230 · Tổng 0.2627  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.4230 → 0.3785, hạng 1 → 2  
**Hình:** `cases/success_1_6160193920_3.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R1:clothing:a blue top –[thuộc]→ ẢNH | R1:clothing:a blue top → ẢNH | +0.0453 | +0.0091 |
| 2 | R1:clothing:a blue top –[nằm phía dưới]→ R2:bodyparts:a giant smile | R1:clothing:a blue top ↔ R2:bodyparts:a giant smile → ẢNH (2 bước) | +0.0009 | +0.0002 |

**Quan hệ không gian mạnh nhất:** R1:clothing:a blue top –[nằm phía dưới]→ R2:bodyparts:a giant smile (Δ = +0.0009)
**Quan sát tự động:**
- Giải thích gồm 2 quan hệ, trong đó 1 quan hệ không gian giữa các vùng và 1 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.0932 (điểm đồ thị 0.4230; sau trộn α=0.2 thành +0.0186 điểm tổng).
- Điểm đồ thị của ảnh CLIP từng xếp đầu thấp hơn ảnh đúng 0.2633; đóng góp cấu trúc của nó -0.0064.

**Nhận xét (nguyên nhân):** Đồ thị kéo ảnh đúng từ hạng 127 lên 1. Đóng góp lớn nhất là vùng áo xanh ("a blue top") được truyền vào nút ảnh (Δ = +0.045), khớp với cụm "blue print shirt" của truy vấn. Quan hệ không gian "áo nằm phía dưới nụ cười" chỉ đóng góp +0.001. Ảnh CLIP từng xếp đầu cũng có vùng áo xanh nhưng điểm đồ thị thấp hơn 0.263, nên bị đẩy xuống. Thành công cần quan hệ thật (no_edge hạng 4, random_edges hạng 13), nhưng phần lớn đến từ cạnh vùng–ảnh. Bỏ 2 quan hệ trong giải thích thì ảnh đúng tụt xuống hạng 2.


## success_2
**Truy vấn:** "A man wearing a green shirt and carrying a shopping bag is reading the newspaper on a city corner while a woman looks on ."  
**Ảnh đúng:** 4980795412  
**Hạng:** CLIP 47 · CLIP+GAT 1 · no_edge 5 · random_edges 4  
**Điểm ảnh đúng:** CLIP 0.2642 · Graph 0.4323 · Tổng 0.2979  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.4323 → 0.3351, hạng 1 → 3  
**Hình:** `cases/success_2_4980795412_1.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R5:clothing:a green shirt –[thuộc]→ ẢNH | R5:clothing:a green shirt → ẢNH | +0.0308 | +0.0062 |
| 2 | R1:people:a man –[thuộc]→ ẢNH | R1:people:a man → ẢNH | +0.0155 | +0.0031 |
| 3 | R1:people:a man –[chồng lấn]→ R5:clothing:a green shirt | R1:people:a man ↔ R5:clothing:a green shirt → ẢNH (2 bước) | +0.0049 | +0.0010 |
| 4 | R5:clothing:a green shirt –[chồng lấn]→ R10:other* | R5:clothing:a green shirt ↔ R10:other* → ẢNH (2 bước) | +0.0043 | +0.0009 |
| 5 | R10:other* –[thuộc]→ ẢNH | R10:other* → ẢNH | +0.0031 | +0.0006 |
| 6 | R5:clothing:a green shirt –[chồng lấn]→ R13:clothing:their shopping bags | R5:clothing:a green shirt ↔ R13:clothing:their shopping bags → ẢNH (2 bước) | +0.0028 | +0.0006 |
| 7 | R13:clothing:their shopping bags –[thuộc]→ ẢNH | R13:clothing:their shopping bags → ẢNH | +0.0027 | +0.0005 |
| 8 | R5:clothing:a green shirt –[chồng lấn]→ R11:clothing* | R5:clothing:a green shirt ↔ R11:clothing* → ẢNH (2 bước) | +0.0025 | +0.0005 |

**Quan hệ không gian mạnh nhất:** R1:people:a man –[chồng lấn]→ R5:clothing:a green shirt (Δ = +0.0049)
**Quan sát tự động:**
- Giải thích gồm 8 quan hệ, trong đó 4 quan hệ không gian giữa các vùng và 4 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.3844 (điểm đồ thị 0.4323; sau trộn α=0.2 thành +0.0769 điểm tổng).
- Điểm đồ thị của ảnh CLIP từng xếp đầu thấp hơn ảnh đúng 0.5093; đóng góp cấu trúc của nó +0.0511.

**Nhận xét (nguyên nhân):** Ca có đóng góp cấu trúc lớn nhất (+0.384): bỏ mọi quan hệ thì điểm đồ thị gần như về 0. Các quan hệ dương khớp đúng ngữ nghĩa truy vấn: vùng "a green shirt" (+0.031), "a man" (+0.016) và "người chồng lấn áo xanh" (+0.005), tức "người mặc áo xanh". Các vùng nền (tòa nhà, lề đường) đóng góp âm. Ảnh CLIP xếp đầu có điểm đồ thị âm (−0.077), nên CLIP + GAT đổi thứ tự. Bỏ 8 quan hệ giải thích thì ảnh đúng tụt xuống hạng 3.


## success_3
**Truy vấn:** "Person looking out of a red colored vehicle driving down the street ."  
**Ảnh đúng:** 3815318559  
**Hạng:** CLIP 33 · CLIP+GAT 1 · no_edge 4 · random_edges 3  
**Điểm ảnh đúng:** CLIP 0.2377 · Graph 0.4056 · Tổng 0.2713  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.4056 → 0.2217, hạng 1 → 7  
**Hình:** `cases/success_3_3815318559_1.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R2:scene:the street –[thuộc]→ ẢNH | R2:scene:the street → ẢNH | +0.0076 | +0.0015 |
| 2 | R4:other:the window –[thuộc]→ ẢNH | R4:other:the window → ẢNH | +0.0072 | +0.0014 |
| 3 | R0:people:A person –[thuộc]→ ẢNH | R0:people:A person → ẢNH | +0.0063 | +0.0013 |
| 4 | R3:clothing:a hat –[chồng lấn]→ R4:other:the window | R3:clothing:a hat ↔ R4:other:the window → ẢNH (2 bước) | +0.0031 | +0.0006 |
| 5 | R0:people:A person –[chồng lấn]→ R3:clothing:a hat | R0:people:A person ↔ R3:clothing:a hat → ẢNH (2 bước) | +0.0021 | +0.0004 |
| 6 | R1:vehicles:a red car –[chồng lấn]→ R4:other:the window | R1:vehicles:a red car ↔ R4:other:the window → ẢNH (2 bước) | +0.0014 | +0.0003 |
| 7 | R0:people:A person –[chồng lấn]→ R1:vehicles:a red car | R0:people:A person ↔ R1:vehicles:a red car → ẢNH (2 bước) | +0.0006 | +0.0001 |
| 8 | R1:vehicles:a red car –[thuộc]→ ẢNH | R1:vehicles:a red car → ẢNH | +0.0004 | +0.0001 |

**Quan hệ không gian mạnh nhất:** R3:clothing:a hat –[chồng lấn]→ R4:other:the window (Δ = +0.0031)
**Quan sát tự động:**
- Giải thích gồm 8 quan hệ, trong đó 4 quan hệ không gian giữa các vùng và 4 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.0989 (điểm đồ thị 0.4056; sau trộn α=0.2 thành +0.0198 điểm tổng).
- Điểm đồ thị của ảnh CLIP từng xếp đầu thấp hơn ảnh đúng 0.5117; đóng góp cấu trúc của nó -0.0947.

**Nhận xét (nguyên nhân):** Đồ thị nâng ảnh từ hạng 33 lên 1 nhờ các vùng người, cửa sổ, đường phố và quan hệ chồng lấn người–mũ–cửa sổ (tương ứng "person looking out of"). Vùng "a red car" không nằm trong giải thích, quan hệ "xe đỏ chồng lấn mũ" còn đóng góp âm: mô hình chưa khai thác thuộc tính màu "red" của truy vấn. Đối thủ (ảnh nhiều xe đỗ, tòa nhà) có đóng góp cấu trúc âm (−0.095), nên một phần thành công đến từ việc đồ thị phạt đối thủ. Bỏ giải thích → hạng 7.


## success_4
**Truy vấn:** "A man in a red shirt holding a red piece of clothing ."  
**Ảnh đúng:** 4821810079  
**Hạng:** CLIP 33 · CLIP+GAT 1 · no_edge 3 · random_edges 13  
**Điểm ảnh đúng:** CLIP 0.2507 · Graph 0.4240 · Tổng 0.2854  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.4240 → 0.3609, hạng 1 → 5  
**Hình:** `cases/success_4_4821810079_2.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R1:clothing:a red shirt –[thuộc]→ ẢNH | R1:clothing:a red shirt → ẢNH | +0.0446 | +0.0089 |
| 2 | R0:people:A man –[chồng lấn]→ R1:clothing:a red shirt | R0:people:A man ↔ R1:clothing:a red shirt → ẢNH (2 bước) | +0.0028 | +0.0006 |

**Quan hệ không gian mạnh nhất:** R0:people:A man –[chồng lấn]→ R1:clothing:a red shirt (Δ = +0.0028)
**Quan sát tự động:**
- Giải thích gồm 2 quan hệ, trong đó 1 quan hệ không gian giữa các vùng và 1 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.2066 (điểm đồ thị 0.4240; sau trộn α=0.2 thành +0.0413 điểm tổng).
- Điểm đồ thị của ảnh CLIP từng xếp đầu thấp hơn ảnh đúng 0.2210; đóng góp cấu trúc của nó +0.0087.

**Nhận xét (nguyên nhân):** Ảnh chỉ có 3 vùng (6 quan hệ). Vùng "a red shirt" đóng góp chính (+0.045), cộng quan hệ "người chồng lấn áo đỏ" (+0.003), khớp "a man in a red shirt". Vùng "a blue market stall" và quan hệ của nó với áo đỏ đóng góp âm. Quan hệ hành động "holding" không có trong đồ thị không gian, nên phần "holding a red piece of clothing" không được giải thích. Bỏ 2 quan hệ giải thích → hạng 5. no_edge chỉ đạt hạng 3.


## success_5
**Truy vấn:** "A male biker jumping a ramp in a contest or crossing the finish line ."  
**Ảnh đúng:** 6225843702  
**Hạng:** CLIP 31 · CLIP+GAT 1 · no_edge 2 · random_edges 2  
**Điểm ảnh đúng:** CLIP 0.2724 · Graph 0.4514 · Tổng 0.3082  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.4514 → 0.4198, hạng 1 → 1  
**Hình:** `cases/success_5_6225843702_1.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R5:other:spandex maneuvers –[thuộc]→ ẢNH | R5:other:spandex maneuvers → ẢNH | +0.0055 | +0.0011 |
| 2 | R2:other:an obstacle –[thuộc]→ ẢNH | R2:other:an obstacle → ẢNH | +0.0055 | +0.0011 |
| 3 | R0:people:A cyclist –[thuộc]→ ẢNH | R0:people:A cyclist → ẢNH | +0.0036 | +0.0007 |
| 4 | R3:vehicles/other:his bike –[thuộc]→ ẢNH | R3:vehicles/other:his bike → ẢNH | +0.0003 | +0.0001 |

**Quan hệ không gian mạnh nhất:** R1:other/scene:an obstacle –[chồng lấn]→ R3:vehicles/other:his bike (Δ = -0.0005)
**Quan sát tự động:**
- Giải thích gồm 4 quan hệ, trong đó 0 quan hệ không gian giữa các vùng và 4 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.0765 (điểm đồ thị 0.4514; sau trộn α=0.2 thành +0.0153 điểm tổng).
- Điểm đồ thị của ảnh CLIP từng xếp đầu thấp hơn ảnh đúng 0.4322; đóng góp cấu trúc của nó -0.0748.

**Nhận xét (nguyên nhân):** Giải thích chỉ gồm 4 cạnh vùng–ảnh (người đạp xe, xe đạp, chướng ngại vật), không có quan hệ không gian nào. Bỏ cả 4 quan hệ thì ảnh vẫn đứng hạng 1, vì đóng góp cấu trúc nhỏ (+0.077) so với điểm từ chính các nút (0.375). Ca này cho thấy giới hạn: thành công không phải lúc nào cũng do quan hệ giữa các vùng, mà chủ yếu do biểu diễn vùng được gộp vào nút ảnh. Đối thủ (có biển báo, xe đạp nằm) có đóng góp cấu trúc âm.


## failure_1 (đồ thị làm hại)
**Truy vấn:** "The photo looks down on a street where five pedestrians are walking ."  
**Ảnh đúng:** 2319389144  
**Hạng:** CLIP 1 · CLIP+GAT 40 · no_edge 24 · random_edges 30  
**Điểm ảnh đúng:** CLIP 0.3031 · Graph -0.0932 · Tổng 0.2238  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị -0.0932 → -0.1297, hạng 40 → 59  
**Hình:** `cases/failure_1_2319389144_1.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R4:people:four people –[thuộc]→ ẢNH | R4:people:four people → ẢNH | +0.0111 | +0.0022 |
| 2 | R5:people:four people –[thuộc]→ ẢNH | R5:people:four people → ẢNH | +0.0047 | +0.0009 |
| 3 | R3:people:four people –[thuộc]→ ẢNH | R3:people:four people → ẢNH | +0.0037 | +0.0007 |
| 4 | R0:scene:the street –[chồng lấn]→ R3:people:four people | R0:scene:the street ↔ R3:people:four people → ẢNH (2 bước) | +0.0019 | +0.0004 |
| 5 | R4:people:four people –[nằm phía dưới]→ R5:people:four people | R4:people:four people ↔ R5:people:four people → ẢNH (2 bước) | +0.0016 | +0.0003 |
| 6 | R0:scene:the street –[chồng lấn]→ R4:people:four people | R0:scene:the street ↔ R4:people:four people → ẢNH (2 bước) | +0.0003 | +0.0001 |
| 7 | R3:people:four people –[nằm bên trái]→ R5:people:four people | R3:people:four people ↔ R5:people:four people → ẢNH (2 bước) | +0.0003 | +0.0001 |

**Quan hệ không gian mạnh nhất:** R0:scene:the street –[chồng lấn]→ R3:people:four people (Δ = +0.0019)
**Quan sát tự động:**
- Giải thích gồm 7 quan hệ, trong đó 4 quan hệ không gian giữa các vùng và 3 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: -0.0136 (điểm đồ thị -0.0932; sau trộn α=0.2 thành -0.0027 điểm tổng).
- Điểm đồ thị của ảnh bị xếp nhầm lên đầu cao hơn ảnh đúng 0.3888; đóng góp cấu trúc của nó +0.0666.

**Nhận xét (nguyên nhân):** Đồ thị làm hại: CLIP hạng 1 → CLIP + GAT hạng 40. Điểm đồ thị của ảnh đúng âm (−0.093) và âm ngay cả khi bỏ mọi quan hệ (−0.080), nên lỗi nằm ở biểu diễn nút của ảnh nhìn từ trên cao với các vùng người rất nhỏ. Ảnh bị xếp nhầm lên đầu có 104 quan hệ, nhiều vùng "cars" và "colorful buildings", điểm đồ thị +0.296. Đồ thị thiên vị ảnh đường phố nhiều vùng hơn là đúng cảnh "looks down on a street".


## failure_2 (đồ thị làm hại)
**Truy vấn:** "Two lovers just met in a date ."  
**Ảnh đúng:** 4686568156  
**Hạng:** CLIP 1 · CLIP+GAT 31 · no_edge 8 · random_edges 20  
**Điểm ảnh đúng:** CLIP 0.2657 · Graph -0.0572 · Tổng 0.2011  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị -0.0572 → -0.0959, hạng 31 → 62  
**Hình:** `cases/failure_2_4686568156_4.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R0:people:A man –[thuộc]→ ẢNH | R0:people:A man → ẢNH | +0.0100 | +0.0020 |
| 2 | R2:clothing:a black jacket –[thuộc]→ ẢNH | R2:clothing:a black jacket → ẢNH | +0.0038 | +0.0008 |
| 3 | R0:people:A man –[chồng lấn]→ R2:clothing:a black jacket | R0:people:A man ↔ R2:clothing:a black jacket → ẢNH (2 bước) | +0.0028 | +0.0006 |
| 4 | R0:people:A man –[chồng lấn]→ R3:clothing:a black jacket | R0:people:A man ↔ R3:clothing:a black jacket → ẢNH (2 bước) | +0.0024 | +0.0005 |
| 5 | R3:clothing:a black jacket –[thuộc]→ ẢNH | R3:clothing:a black jacket → ẢNH | +0.0019 | +0.0004 |
| 6 | R0:people:A man –[chồng lấn]→ R1:people:couple | R0:people:A man ↔ R1:people:couple → ẢNH (2 bước) | +0.0016 | +0.0003 |
| 7 | R2:clothing:a black jacket –[chồng lấn]→ R5:clothing:earrings | R2:clothing:a black jacket ↔ R5:clothing:earrings → ẢNH (2 bước) | +0.0010 | +0.0002 |
| 8 | R0:people:A man –[nằm bên trái]→ R5:clothing:earrings | R0:people:A man ↔ R5:clothing:earrings → ẢNH (2 bước) | +0.0003 | +0.0001 |

**Quan hệ không gian mạnh nhất:** R0:people:A man –[chồng lấn]→ R2:clothing:a black jacket (Δ = +0.0028)
**Quan sát tự động:**
- Giải thích gồm 8 quan hệ, trong đó 5 quan hệ không gian giữa các vùng và 3 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: -0.0194 (điểm đồ thị -0.0572; sau trộn α=0.2 thành -0.0039 điểm tổng).
- Điểm đồ thị của ảnh bị xếp nhầm lên đầu cao hơn ảnh đúng 0.2609; đóng góp cấu trúc của nó +0.0075.

**Nhận xét (nguyên nhân):** Truy vấn trừu tượng ("two lovers just met in a date"): không vùng hay quan hệ không gian nào biểu diễn được khái niệm "hẹn hò". Vùng "a woman's face" đóng góp âm (−0.010) và kéo theo quan hệ chồng lấn áo khoác–mặt (−0.006). Điểm đồ thị ảnh đúng âm (−0.057), trong khi ảnh nhầm có cặp "man chồng lấn woman" và trang phục đỏ. Đây là loại truy vấn gián tiếp mà đồ thị không gian không hỗ trợ được, cần quan hệ ngữ nghĩa (cấp độ 2–3).


## failure_3 (đồ thị làm hại)
**Truy vấn:** "The young man is waiting with others on the sidewalk ."  
**Ảnh đúng:** 2735558076  
**Hạng:** CLIP 1 · CLIP+GAT 20 · no_edge 13 · random_edges 23  
**Điểm ảnh đúng:** CLIP 0.3018 · Graph 0.0065 · Tổng 0.2427  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.0065 → -0.0329, hạng 20 → 37  
**Hình:** `cases/failure_3_2735558076_3.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R9:clothing:warm clothes –[thuộc]→ ẢNH | R9:clothing:warm clothes → ẢNH | +0.0067 | +0.0013 |
| 2 | R8:clothing:warm clothes –[thuộc]→ ẢNH | R8:clothing:warm clothes → ẢNH | +0.0065 | +0.0013 |
| 3 | R1:people:A group of people –[thuộc]→ ẢNH | R1:people:A group of people → ẢNH | +0.0043 | +0.0009 |
| 4 | R0:people:a boy –[thuộc]→ ẢNH | R0:people:a boy → ẢNH | +0.0025 | +0.0005 |
| 5 | R7:bodyparts:A row of legs –[chồng lấn]→ R8:clothing:warm clothes | R7:bodyparts:A row of legs ↔ R8:clothing:warm clothes → ẢNH (2 bước) | +0.0020 | +0.0004 |
| 6 | R0:people:a boy –[chồng lấn]→ R7:bodyparts:A row of legs | R0:people:a boy ↔ R7:bodyparts:A row of legs → ẢNH (2 bước) | +0.0018 | +0.0004 |
| 7 | R1:people:A group of people –[chồng lấn]→ R7:bodyparts:A row of legs | R1:people:A group of people ↔ R7:bodyparts:A row of legs → ẢNH (2 bước) | +0.0017 | +0.0003 |
| 8 | R1:people:A group of people –[chồng lấn]→ R9:clothing:warm clothes | R1:people:A group of people ↔ R9:clothing:warm clothes → ẢNH (2 bước) | +0.0017 | +0.0003 |

**Quan hệ không gian mạnh nhất:** R7:bodyparts:A row of legs –[chồng lấn]→ R8:clothing:warm clothes (Δ = +0.0020)
**Quan sát tự động:**
- Giải thích gồm 8 quan hệ, trong đó 4 quan hệ không gian giữa các vùng và 4 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: -0.0191 (điểm đồ thị 0.0065; sau trộn α=0.2 thành -0.0038 điểm tổng).
- Điểm đồ thị của ảnh bị xếp nhầm lên đầu cao hơn ảnh đúng 0.3721; đóng góp cấu trúc của nó +0.1138.

**Nhận xét (nguyên nhân):** CLIP hạng 1 → GAT hạng 20. Vùng "pavement" (khớp "sidewalk") lại đóng góp âm (−0.005), vùng "black boots" âm (−0.007). Ảnh đúng có 44 quan hệ nhưng đóng góp cấu trúc âm (−0.019). Ảnh nhầm (một cặp đôi, áo, quần short) có đóng góp cấu trúc +0.114 nhờ các vùng trang phục rõ ràng. Mô hình ưu tiên vùng quần áo hơn bối cảnh "đang chờ trên vỉa hè".


## failure_4 (không cứu được)
**Truy vấn:** "Man with helmet performing a trick while rollerblading ."  
**Ảnh đúng:** 2877088081  
**Hạng:** CLIP 82 · CLIP+GAT 2 · no_edge 1 · random_edges 2  
**Điểm ảnh đúng:** CLIP 0.2461 · Graph 0.5514 · Tổng 0.3072  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.5514 → 0.4970, hạng 2 → 3  
**Hình:** `cases/failure_4_2877088081_2.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R3:clothing* –[thuộc]→ ẢNH | R3:clothing* → ẢNH | +0.0188 | +0.0038 |
| 2 | R2:clothing:a black shirt –[thuộc]→ ẢNH | R2:clothing:a black shirt → ẢNH | +0.0034 | +0.0007 |
| 3 | R2:clothing:a black shirt –[chồng lấn]→ R3:clothing* | R2:clothing:a black shirt ↔ R3:clothing* → ẢNH (2 bước) | +0.0021 | +0.0004 |
| 4 | R1:other:a stair rail –[chồng lấn]→ R3:clothing* | R1:other:a stair rail ↔ R3:clothing* → ẢNH (2 bước) | +0.0013 | +0.0003 |
| 5 | R3:clothing* –[nằm phía trên]→ R4:clothing:jeans | R3:clothing* ↔ R4:clothing:jeans → ẢNH (2 bước) | +0.0007 | +0.0001 |
| 6 | R1:other:a stair rail –[chồng lấn]→ R2:clothing:a black shirt | R1:other:a stair rail ↔ R2:clothing:a black shirt → ẢNH (2 bước) | +0.0004 | +0.0001 |
| 7 | R2:clothing:a black shirt –[chồng lấn]→ R4:clothing:jeans | R2:clothing:a black shirt ↔ R4:clothing:jeans → ẢNH (2 bước) | +0.0003 | +0.0001 |
| 8 | R1:other:a stair rail –[chồng lấn]→ R4:clothing:jeans | R1:other:a stair rail ↔ R4:clothing:jeans → ẢNH (2 bước) | +0.0003 | +0.0001 |

**Quan hệ không gian mạnh nhất:** R2:clothing:a black shirt –[chồng lấn]→ R3:clothing* (Δ = +0.0021)
**Quan sát tự động:**
- Giải thích gồm 8 quan hệ, trong đó 6 quan hệ không gian giữa các vùng và 2 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.2073 (điểm đồ thị 0.5514; sau trộn α=0.2 thành +0.0415 điểm tổng).
- Điểm đồ thị của ảnh bị xếp nhầm lên đầu thấp hơn ảnh đúng 0.1959; đóng góp cấu trúc của nó +0.2097.

**Nhận xét (nguyên nhân):** Suýt đúng: đồ thị nâng ảnh từ hạng 82 lên 2 (điểm đồ thị 0.551, cao hơn ảnh xếp đầu 0.196) nhưng CLIP của ảnh xếp đầu (trượt ván, kính râm, mặt đường) cao hơn nên với α = 0.2 vẫn thắng. Vùng đóng góp chính là vùng mũ bảo hiểm (theo annotation, chỉ được nhắc trong chính câu truy vấn nên hình chỉ ghi loại "clothing*"); vùng "A boy" đóng góp âm. no_edge xếp đúng hạng 1: ở ca này quan hệ thật không giúp thêm.


## failure_5 (không cứu được)
**Truy vấn:** "A woman gives the 'thumbs up ' before her legs are amputated ."  
**Ảnh đúng:** 1775223211  
**Hạng:** CLIP 59 · CLIP+GAT 2 · no_edge 4 · random_edges 1  
**Điểm ảnh đúng:** CLIP 0.2353 · Graph 0.3891 · Tổng 0.2661  
**Bỏ các quan hệ trong giải thích:** điểm đồ thị 0.3891 → 0.3322, hạng 2 → 5  
**Hình:** `cases/failure_5_1775223211_2.png`

| # | Quan hệ | Đường đi | Δ điểm đồ thị | Δ điểm tổng |
|---|---|---|---:|---:|
| 1 | R6:other* –[thuộc]→ ẢNH | R6:other* → ẢNH | +0.0174 | +0.0035 |
| 2 | R5:other:cloth –[thuộc]→ ẢNH | R5:other:cloth → ẢNH | +0.0128 | +0.0026 |
| 3 | R5:other:cloth –[chồng lấn]→ R6:other* | R5:other:cloth ↔ R6:other* → ẢNH (2 bước) | +0.0047 | +0.0009 |
| 4 | R7:bodyparts* –[thuộc]→ ẢNH | R7:bodyparts* → ẢNH | +0.0027 | +0.0005 |
| 5 | R5:other:cloth –[chồng lấn]→ R7:bodyparts* | R5:other:cloth ↔ R7:bodyparts* → ẢNH (2 bước) | +0.0020 | +0.0004 |
| 6 | R1:clothing:a ball cap –[thuộc]→ ẢNH | R1:clothing:a ball cap → ẢNH | +0.0009 | +0.0002 |
| 7 | R1:clothing:a ball cap –[nằm phía trên]→ R6:other* | R1:clothing:a ball cap ↔ R6:other* → ẢNH (2 bước) | +0.0007 | +0.0001 |
| 8 | R2:people:One woman –[chồng lấn]→ R7:bodyparts* | R2:people:One woman ↔ R7:bodyparts* → ẢNH (2 bước) | +0.0007 | +0.0001 |

**Quan hệ không gian mạnh nhất:** R5:other:cloth –[chồng lấn]→ R6:other* (Δ = +0.0047)
**Quan sát tự động:**
- Giải thích gồm 8 quan hệ, trong đó 4 quan hệ không gian giữa các vùng và 4 quan hệ vùng–ảnh.
- Đóng góp cấu trúc vào điểm đồ thị của ảnh đúng: +0.1698 (điểm đồ thị 0.3891; sau trộn α=0.2 thành +0.0340 điểm tổng).
- Điểm đồ thị của ảnh bị xếp nhầm lên đầu thấp hơn ảnh đúng 0.1869; đóng góp cấu trúc của nó +0.1841.

**Nhận xét (nguyên nhân):** Suýt đúng: hạng CLIP 59 → 2 (random_edges lại xếp đúng hạng 1). Đóng góp chính đến từ vùng "the thumbs up" (theo annotation, chỉ có trong câu truy vấn nên hình ghi "other*") và vùng "cloth"; các vùng "the unfinished item", "a loom" đóng góp âm. Ảnh nhầm cũng có người phụ nữ, vùng "her hand" và vùng "something" với đóng góp cấu trúc lớn (+0.184). Nội dung đặc thù của truy vấn ("thumbs up", bối cảnh y tế) không có quan hệ tương ứng trong đồ thị không gian.
