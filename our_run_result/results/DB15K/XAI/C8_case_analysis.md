# DB15K — 5 truy vấn thành công và 5 truy vấn thất bại (seed 2027, K = 2)


## thanh_cong — q_6f081eb3d974c9508fe6ab34
**Truy vấn:** Find an image of the entity that is related to Ronald Ernest "Ron" Paul by party.  
**Hạng ảnh đúng:** CLIP 105 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_6f081eb3d974c9508fe6ab34.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Libertarian Party (United States)** · điểm đồ thị -1.009 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.87, ảnh 0.07, văn bản 0.06; che từng phương thức: cấu trúc (KG) Δ +49.873 (hạng → 1000), ảnh Δ +0.208 (hạng → 1), văn bản Δ +1.266 (hạng → 7)
- Bộ ba train của thực thể: 8 · điểm gốc -1.009 · bản học lại đủ bộ ba -0.415

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Ronald Ernest "Ron" Paul –[child]→ Libertarian Party (United States) | Ronald Ernest "Ron" Paul → Libertarian Party (United States) (1 bước) | +0.4891 |
| 2 | Libertarian Party (United States) –[type]→ Island | Ronald Ernest "Ron" Paul → Island → Libertarian Party (United States) (2 bước) | +0.2866 |

Bỏ đồng thời top-2: điểm giảm +0.5170, hạng thay đổi +0.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.2964, +0.0

## thanh_cong — q_ffbbb7ed71136914b34d1caf
**Truy vấn:** Find an image of the entity whose influenced by relation points to Richard Dawkins.  
**Hạng ảnh đúng:** CLIP 55 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_ffbbb7ed71136914b34d1caf.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Christopher Hitchens** · điểm đồ thị -0.788 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.91, ảnh 0.04, văn bản 0.05; che từng phương thức: cấu trúc (KG) Δ +74.829 (hạng → 1000), ảnh Δ -0.147 (hạng → 1), văn bản Δ +0.195 (hạng → 1)
- Bộ ba train của thực thể: 28 · điểm gốc -0.788 · bản học lại đủ bộ ba -0.333

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Christopher Hitchens –[influenced]→ Richard Dawkins | Richard Dawkins → Christopher Hitchens (1 bước) | +0.1982 |
| 2 | Christopher Hitchens –[influenced by]→ Albert Camus | Albert Camus → Christopher Hitchens (lân cận (không nối về anchor)) | +0.1291 |

Bỏ đồng thời top-2: điểm giảm +0.3248, hạng thay đổi +0.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0201, +0.0

## thanh_cong — q_04726fae7096a9c43e1691dd
**Truy vấn:** Find an image of the entity whose associated band relation points to Missy "Misdemeanor" Elliot.  
**Hạng ảnh đúng:** CLIP 624 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_04726fae7096a9c43e1691dd.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Timbaland** · điểm đồ thị +1.938 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.91, ảnh 0.05, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +71.015 (hạng → 1000), ảnh Δ +0.344 (hạng → 1), văn bản Δ +1.326 (hạng → 2)
- Bộ ba train của thực thể: 33 · điểm gốc +1.938 · bản học lại đủ bộ ba +2.233

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Timbaland –[associated musical artist]→ Missy "Misdemeanor" Elliot | Missy "Misdemeanor" Elliot → Timbaland (1 bước) | +0.3352 |
| 2 | Timbaland –[associated band]→ Static Major | Missy "Misdemeanor" Elliot → Static Major → Timbaland (2 bước) | +0.1581 |

Bỏ đồng thời top-2: điểm giảm +0.6658, hạng thay đổi +0.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): -0.0537, +0.0

## thanh_cong — q_919aeff1938ffdb3111bd4a7
**Truy vấn:** Find an image of the entity that is related to Iraklis 1908 Thessaloniki F.C. by ground.  
**Hạng ảnh đúng:** CLIP 106 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_919aeff1938ffdb3111bd4a7.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Thessaloniki** · điểm đồ thị -1.339 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.88, ảnh 0.04, văn bản 0.08; che từng phương thức: cấu trúc (KG) Δ +53.249 (hạng → 1000), ảnh Δ -0.134 (hạng → 1), văn bản Δ +1.462 (hạng → 14)
- Bộ ba train của thực thể: 6 · điểm gốc -1.339 · bản học lại đủ bộ ba -0.907

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Thessaloniki –[time zone]→ Eastern European Time | Eastern European Time → Thessaloniki (lân cận (không nối về anchor)) | +0.9003 |
| 2 | Thessaloniki –[see also]→ Greece | Iraklis 1908 Thessaloniki F.C. → Greece → Thessaloniki (2 bước) | +0.6789 |

Bỏ đồng thời top-2: điểm giảm +1.9996, hạng thay đổi +31.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.5605, +0.3

## thanh_cong — q_e6accc926397a1bd8e4067cc
**Truy vấn:** Find an image of the entity that is related to Oklahoma! (1955 film) by music composer.  
**Hạng ảnh đúng:** CLIP 117 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_e6accc926397a1bd8e4067cc.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Richard Rodgers** · điểm đồ thị -1.032 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.92, ảnh 0.04, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +75.520 (hạng → 1000), ảnh Δ +0.060 (hạng → 1), văn bản Δ +1.075 (hạng → 9)
- Bộ ba train của thực thể: 8 · điểm gốc -1.032 · bản học lại đủ bộ ba -0.793

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | The Sound of Music (film) –[music composer]→ Richard Rodgers | The Sound of Music (film) → Richard Rodgers (lân cận (không nối về anchor)) | +0.4279 |
| 2 | Richard Rodgers –[type]→ Island | Island → Richard Rodgers (lân cận (không nối về anchor)) | +0.3124 |

Bỏ đồng thời top-2: điểm giảm +0.9362, hạng thay đổi +7.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.1183, +1.2

## that_bai_bi_lam_hai — q_668a536d1098095bb788a0b9
**Truy vấn:** Find an image of the entity whose see also relation points to Streamline Moderne.  
**Hạng ảnh đúng:** CLIP 1 → CLIP + AdaMF 70  
**Hình:** `cases/that_bai_bi_lam_hai_q_668a536d1098095bb788a0b9.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Art Deco** · điểm đồ thị -7.856 · hạng 70
- Phương thức (E2): attention cấu trúc (KG) 0.89, ảnh 0.06, văn bản 0.05; che từng phương thức: cấu trúc (KG) Δ +41.871 (hạng → 1000), ảnh Δ +0.543 (hạng → 246), văn bản Δ +0.905 (hạng → 462)
- Bộ ba train của thực thể: 6 · điểm gốc -7.856 · bản học lại đủ bộ ba -7.854

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Art Deco –[see also]→ United States | United States → Art Deco (lân cận (không nối về anchor)) | +0.4660 |
| 2 | Art Deco –[see also]→ Sydney | Sydney → Art Deco (lân cận (không nối về anchor)) | +0.1881 |

Bỏ đồng thời top-2: điểm giảm +1.0112, hạng thay đổi +451.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.2967, +86.2

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Modern architecture** · điểm đồ thị -6.296 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.90, ảnh 0.04, văn bản 0.06; che từng phương thức: cấu trúc (KG) Δ +47.794 (hạng → 1000), ảnh Δ +0.125 (hạng → 1), văn bản Δ +0.906 (hạng → 9)
- Bộ ba train của thực thể: 1 · điểm gốc -6.296 · bản học lại đủ bộ ba -8.371

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|

## that_bai_bi_lam_hai — q_4db1993eb9b24bd0c441f766
**Truy vấn:** Find an image of the entity whose starring relation points to Rock Hudson.  
**Hạng ảnh đúng:** CLIP 1 → CLIP + AdaMF 4  
**Hình:** `cases/that_bai_bi_lam_hai_q_4db1993eb9b24bd0c441f766.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Pillow Talk (film)** · điểm đồ thị -1.700 · hạng 4
- Phương thức (E2): attention cấu trúc (KG) 0.91, ảnh 0.05, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +72.973 (hạng → 1000), ảnh Δ +0.178 (hạng → 10), văn bản Δ +0.873 (hạng → 90)
- Bộ ba train của thực thể: 4 · điểm gốc -1.700 · bản học lại đủ bộ ba -1.075

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Rock Hudson –[known for]→ Pillow Talk (film) | Rock Hudson → Pillow Talk (film) (1 bước) | +1.0789 |
| 2 | Pillow Talk (film) –[distributor]→ Universal Studios | Universal Studios → Pillow Talk (film) (lân cận (không nối về anchor)) | +0.3759 |

Bỏ đồng thời top-2: điểm giảm +2.1475, hạng thay đổi +343.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +1.0788, +158.7

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Dynasty (TV series)** · điểm đồ thị -1.373 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.92, ảnh 0.04, văn bản 0.05; che từng phương thức: cấu trúc (KG) Δ +74.318 (hạng → 1000), ảnh Δ +0.056 (hạng → 2), văn bản Δ +0.977 (hạng → 50)
- Bộ ba train của thực thể: 9 · điểm gốc -1.373 · bản học lại đủ bộ ba -0.758

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Rock Hudson –[known for]→ Dynasty (TV series) | Rock Hudson → Dynasty (TV series) (1 bước) | +0.8841 |
| 2 | Dynasty (TV series) –[type]→ Island | Rock Hudson → Island → Dynasty (TV series) (2 bước) | +0.0711 |

Bỏ đồng thời top-2: điểm giảm +0.6776, hạng thay đổi +18.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.3314, +13.2

## that_bai_bi_lam_hai — q_7aff36a270190fc77bfc4c72
**Truy vấn:** Find an image of the entity that is related to Soul music by music fusion genre.  
**Hạng ảnh đúng:** CLIP 1 → CLIP + AdaMF 4  
**Hình:** `cases/that_bai_bi_lam_hai_q_7aff36a270190fc77bfc4c72.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Soul jazz** · điểm đồ thị +0.169 · hạng 4
- Phương thức (E2): attention cấu trúc (KG) 0.89, ảnh 0.06, văn bản 0.05; che từng phương thức: cấu trúc (KG) Δ +54.377 (hạng → 1000), ảnh Δ +0.252 (hạng → 4), văn bản Δ +0.811 (hạng → 7)
- Bộ ba train của thực thể: 12 · điểm gốc +0.169 · bản học lại đủ bộ ba +1.243

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Soul jazz –[stylistic origin]→ Soul music | Soul music → Soul jazz (1 bước) | +0.4180 |
| 2 | Soul jazz –[instrument]→ Bass guitar | Bass guitar → Soul jazz (lân cận (không nối về anchor)) | +0.1306 |

Bỏ đồng thời top-2: điểm giảm +0.6890, hạng thay đổi +3.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.1267, +0.6

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Funk** · điểm đồ thị +0.668 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.92, văn bản 0.08; che từng phương thức: cấu trúc (KG) Δ +65.631 (hạng → 1000), văn bản Δ +0.516 (hạng → 4)
- Bộ ba train của thực thể: 106 · điểm gốc +0.668 · bản học lại đủ bộ ba +0.973

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Funk –[stylistic origin]→ Soul music | Soul music → Funk (1 bước) | +0.0776 |
| 2 | Soul music –[derivative]→ Funk | Soul music → Funk (1 bước) | +0.0467 |

Bỏ đồng thời top-2: điểm giảm +0.1438, hạng thay đổi +0.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): -0.0021, +0.0

## that_bai_khong_cuu_duoc — q_49422bb19033699e0e281b9d
**Truy vấn:** Find an image of the entity whose death place relation points to Danbury, Connecticut.  
**Hạng ảnh đúng:** CLIP 361 → CLIP + AdaMF 170  
**Hình:** `cases/that_bai_khong_cuu_duoc_q_49422bb19033699e0e281b9d.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Maurice Sendak** · điểm đồ thị -4.589 · hạng 170
- Phương thức (E2): attention cấu trúc (KG) 0.92, ảnh 0.05, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +65.439 (hạng → 1000), ảnh Δ -0.090 (hạng → 145), văn bản Δ -0.047 (hạng → 159)
- Bộ ba train của thực thể: 5 · điểm gốc -4.589 · bản học lại đủ bộ ba -4.067

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Maurice Sendak –[type]→ Island | Island → Maurice Sendak (lân cận (không nối về anchor)) | +0.6916 |
| 2 | Stephen Colbert –[influenced by]→ Maurice Sendak | Stephen Colbert → Maurice Sendak (lân cận (không nối về anchor)) | +0.5381 |

Bỏ đồng thời top-2: điểm giảm +1.8712, hạng thay đổi +757.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +1.0208, +468.9

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Mario Puzo** · điểm đồ thị -3.123 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.93, ảnh 0.04, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +73.561 (hạng → 1000), ảnh Δ -0.064 (hạng → 1), văn bản Δ +0.012 (hạng → 1)
- Bộ ba train của thực thể: 11 · điểm gốc -3.123 · bản học lại đủ bộ ba -2.745

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Christopher Columbus: The Discovery –[writer]→ Mario Puzo | Christopher Columbus: The Discovery → Mario Puzo (lân cận (không nối về anchor)) | +0.2477 |
| 2 | Mario Puzo –[birth place]→ New York | New York → Mario Puzo (lân cận (không nối về anchor)) | +0.0589 |

Bỏ đồng thời top-2: điểm giảm +0.2642, hạng thay đổi +6.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0727, +3.6

## that_bai_khong_cuu_duoc — q_ecfc6d0600ab8e1ebda614fa
**Truy vấn:** Find an image of the entity that is related to Arthur C. Clarke by influenced.  
**Hạng ảnh đúng:** CLIP 19 → CLIP + AdaMF 366  
**Hình:** `cases/that_bai_khong_cuu_duoc_q_ecfc6d0600ab8e1ebda614fa.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Stephen Baxter** · điểm đồ thị -3.665 · hạng 366
- Phương thức (E2): attention cấu trúc (KG) 0.92, ảnh 0.04, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +69.859 (hạng → 1000), ảnh Δ -0.056 (hạng → 324), văn bản Δ +0.063 (hạng → 394)
- Bộ ba train của thực thể: 3 · điểm gốc -3.665 · bản học lại đủ bộ ba -2.915

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Stephen Baxter –[type]→ Governor | Governor → Stephen Baxter (lân cận (không nối về anchor)) | +1.3649 |
| 2 | Stephen Baxter –[genre]→ Alternate history | Alternate history → Stephen Baxter (lân cận (không nối về anchor)) | +0.6727 |

Bỏ đồng thời top-2: điểm giảm +4.0078, hạng thay đổi +632.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +1.8987, +378.1

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Arthur C. Clarke** · điểm đồ thị +0.743 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.91, ảnh 0.05, văn bản 0.04; che từng phương thức: cấu trúc (KG) Δ +65.131 (hạng → 1000), ảnh Δ +0.161 (hạng → 1), văn bản Δ +1.313 (hạng → 1)
- Bộ ba train của thực thể: 11 · điểm gốc +0.743 · bản học lại đủ bộ ba +0.739

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Arthur C. Clarke –[type]→ Planet | Planet → Arthur C. Clarke (lân cận (không nối về anchor)) | +0.2728 |
| 2 | Arthur C. Clarke –[influenced by]→ Jules Verne | Jules Verne → Arthur C. Clarke (lân cận (không nối về anchor)) | +0.2053 |

Bỏ đồng thời top-2: điểm giảm +0.6727, hạng thay đổi +0.0 · bỏ 2 bộ ba ngẫu nhiên (trung bình 10 lần): +0.1751, +0.0