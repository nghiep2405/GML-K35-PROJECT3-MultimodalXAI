# FB15K — 5 truy vấn thành công và 5 truy vấn thất bại (seed 2027, K = 8)


## thanh_cong — q_ddb73cad385ebf35f7137e30
**Truy vấn:** Find an image of the entity that is related to Rick Rubin by nationality.  
**Hạng ảnh đúng:** CLIP 269 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_ddb73cad385ebf35f7137e30.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **United States of America** · điểm đồ thị -0.179 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.42, ảnh 0.16, văn bản 0.42; che từng phương thức: cấu trúc (KG) Δ +0.180 (hạng → 720), ảnh Δ -0.015 (hạng → 1), văn bản Δ -0.018 (hạng → 1)
- Bộ ba train của thực thể: 7614 · điểm gốc -0.179 · bản học lại đủ bộ ba -0.046

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Jay-Z –[person / nationality]→ United States of America | Rick Rubin → Jay-Z → United States of America (2 bước) | +0.0006 |
| 2 | Dan Wilson –[person / nationality]→ United States of America | Rick Rubin → Dan Wilson → United States of America (2 bước) | +0.0005 |
| 3 | Brian Henson –[person / nationality]→ United States of America | Brian Henson → United States of America (lân cận (không nối về anchor)) | +0.0004 |
| 4 | Johnny Cash –[person / nationality]→ United States of America | Rick Rubin → Johnny Cash → United States of America (2 bước) | +0.0004 |
| 5 | Dennis Hopper –[person / nationality]→ United States of America | Dennis Hopper → United States of America (lân cận (không nối về anchor)) | +0.0004 |
| 6 | Irwin Winkler –[person / nationality]→ United States of America | Irwin Winkler → United States of America (lân cận (không nối về anchor)) | +0.0003 |
| 7 | Cynthia Nixon –[person / nationality]→ United States of America | Cynthia Nixon → United States of America (lân cận (không nối về anchor)) | +0.0003 |
| 8 | Jeff Probst –[person / nationality]→ United States of America | Jeff Probst → United States of America (lân cận (không nối về anchor)) | +0.0003 |

Bỏ đồng thời top-8: điểm giảm +0.0036, hạng thay đổi +0.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0000, +0.0

## thanh_cong — q_a0bf025f4dd34598a476354c
**Truy vấn:** Find an image of the entity that is related to Handball by country.  
**Hạng ảnh đúng:** CLIP 20 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_a0bf025f4dd34598a476354c.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Poland** · điểm đồ thị -0.095 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.32, ảnh 0.16, văn bản 0.53; che từng phương thức: cấu trúc (KG) Δ +0.110 (hạng → 198), ảnh Δ +0.001 (hạng → 1), văn bản Δ +0.108 (hạng → 189)
- Bộ ba train của thực thể: 449 · điểm gốc -0.095 · bản học lại đủ bộ ba -0.038

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Poland –[olympic medal honor / olympics]→ 2008 Summer Olympics | Handball → 2008 Summer Olympics → Poland (2 bước) | +0.0001 |
| 2 | Poland –[olympic medal honor / olympics]→ 1972 Summer Olympics | Handball → 1972 Summer Olympics → Poland (2 bước) | +0.0001 |
| 3 | Poland –[olympic medal honor / olympics]→ 1988 Summer Olympics | Handball → 1988 Summer Olympics → Poland (2 bước) | +0.0000 |
| 4 | Poland –[olympic medal honor / olympics]→ 1980 Summer Olympics | Handball → 1980 Summer Olympics → Poland (2 bước) | +0.0000 |
| 5 | 2008 Summer Olympics –[olympic games / participating countries]→ Poland | Handball → 2008 Summer Olympics → Poland (2 bước) | +0.0000 |
| 6 | Poland –[olympic medal honor / olympics]→ 2000 Summer Olympics | Handball → 2000 Summer Olympics → Poland (2 bước) | +0.0000 |
| 7 | Poland –[olympic medal honor / olympics]→ 1996 Summer Olympics | Handball → 1996 Summer Olympics → Poland (2 bước) | +0.0000 |
| 8 | Poland –[olympic athlete affiliation / olympics]→ 2008 Summer Olympics | Handball → 2008 Summer Olympics → Poland (2 bước) | +0.0000 |

Bỏ đồng thời top-8: điểm giảm +0.0005, hạng thay đổi +0.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0007, +0.0

## thanh_cong — q_073c81175398d057e8ea54fc
**Truy vấn:** Find an image of the entity that is related to Gran Torino by language.  
**Hạng ảnh đúng:** CLIP 409 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_073c81175398d057e8ea54fc.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **English Language** · điểm đồ thị -0.125 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.38, ảnh 0.16, văn bản 0.46; che từng phương thức: cấu trúc (KG) Δ +0.192 (hạng → 695), ảnh Δ -0.009 (hạng → 1), văn bản Δ +0.000 (hạng → 1)
- Bộ ba train của thực thể: 2675 · điểm gốc -0.125 · bản học lại đủ bộ ba -0.026

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | The Pink Panther –[film / language]→ English Language | The Pink Panther → English Language (lân cận (không nối về anchor)) | +0.0001 |
| 2 | Midnight Express –[film / language]→ English Language | Midnight Express → English Language (lân cận (không nối về anchor)) | +0.0001 |
| 3 | The Firm –[film / language]→ English Language | The Firm → English Language (lân cận (không nối về anchor)) | +0.0001 |
| 4 | The Newton Boys –[film / language]→ English Language | The Newton Boys → English Language (lân cận (không nối về anchor)) | +0.0000 |
| 5 | Robin Hood –[film / language]→ English Language | Robin Hood → English Language (lân cận (không nối về anchor)) | +0.0000 |
| 6 | 28 Days Later –[film / language]→ English Language | 28 Days Later → English Language (lân cận (không nối về anchor)) | +0.0000 |
| 7 | The Legend of Bagger Vance –[film / language]→ English Language | The Legend of Bagger Vance → English Language (lân cận (không nối về anchor)) | +0.0000 |
| 8 | The Dark Knight –[film / language]→ English Language | The Dark Knight → English Language (lân cận (không nối về anchor)) | +0.0000 |

Bỏ đồng thời top-8: điểm giảm +0.0004, hạng thay đổi +0.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): -0.0002, +0.0

## thanh_cong — q_9adcf114103c79868a0d57d8
**Truy vấn:** Find an image of the entity that is related to Demi Moore by profession.  
**Hạng ảnh đúng:** CLIP 95 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_9adcf114103c79868a0d57d8.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Actor-GB** · điểm đồ thị -0.136 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.47, ảnh 0.13, văn bản 0.41; che từng phương thức: cấu trúc (KG) Δ +0.135 (hạng → 275), ảnh Δ +0.017 (hạng → 2), văn bản Δ +0.051 (hạng → 10)
- Bộ ba train của thực thể: 2272 · điểm gốc -0.136 · bản học lại đủ bộ ba -0.012

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Sharon Stone –[person / profession]→ Actor-GB | Demi Moore → Sharon Stone → Actor-GB (2 bước) | +0.0004 |
| 2 | Alicia Keys –[person / profession]→ Actor-GB | Demi Moore → Alicia Keys → Actor-GB (2 bước) | +0.0004 |
| 3 | Colin Farrell –[person / profession]→ Actor-GB | Demi Moore → Colin Farrell → Actor-GB (2 bước) | +0.0003 |
| 4 | Joy Bryant –[person / profession]→ Actor-GB | Demi Moore → Joy Bryant → Actor-GB (2 bước) | +0.0003 |
| 5 | Kate Hudson –[person / profession]→ Actor-GB | Demi Moore → Kate Hudson → Actor-GB (2 bước) | +0.0003 |
| 6 | Robert Duvall –[person / profession]→ Actor-GB | Demi Moore → Robert Duvall → Actor-GB (2 bước) | +0.0003 |
| 7 | Helen Hunt –[person / profession]→ Actor-GB | Demi Moore → Helen Hunt → Actor-GB (2 bước) | +0.0002 |
| 8 | Brad Pitt –[person / profession]→ Actor-GB | Demi Moore → Brad Pitt → Actor-GB (2 bước) | +0.0002 |

Bỏ đồng thời top-8: điểm giảm +0.0024, hạng thay đổi +0.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0004, +0.0

## thanh_cong — q_dbf5a485b65e5fc618cd1f17
**Truy vấn:** Find an image of the entity that is related to Magnolia Pictures by citytown.  
**Hạng ảnh đúng:** CLIP 364 → CLIP + AdaMF 1  
**Hình:** `cases/thanh_cong_q_dbf5a485b65e5fc618cd1f17.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **New York City** · điểm đồ thị -0.115 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.38, ảnh 0.18, văn bản 0.44; che từng phương thức: cấu trúc (KG) Δ +0.160 (hạng → 695), ảnh Δ -0.003 (hạng → 1), văn bản Δ +0.019 (hạng → 2)
- Bộ ba train của thực thể: 1077 · điểm gốc -0.115 · bản học lại đủ bộ ba -0.068

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Koch Entertainment –[mailing address / citytown]→ New York City | Koch Entertainment → New York City (lân cận (không nối về anchor)) | +0.0011 |
| 2 | Sony Music Entertainment –[mailing address / citytown]→ New York City | Sony Music Entertainment → New York City (lân cận (không nối về anchor)) | +0.0011 |
| 3 | Associated Press –[mailing address / citytown]→ New York City | Associated Press → New York City (lân cận (không nối về anchor)) | +0.0011 |
| 4 | Jujamcyn Theaters –[mailing address / citytown]→ New York City | Jujamcyn Theaters → New York City (lân cận (không nối về anchor)) | +0.0011 |
| 5 | Lifetime Television –[mailing address / citytown]→ New York City | Lifetime Television → New York City (lân cận (không nối về anchor)) | +0.0011 |
| 6 | Warner Music Group –[mailing address / citytown]→ New York City | Warner Music Group → New York City (lân cận (không nối về anchor)) | +0.0011 |
| 7 | Citigroup –[mailing address / citytown]→ New York City | Citigroup → New York City (lân cận (không nối về anchor)) | +0.0010 |
| 8 | Sony Pictures Classics –[mailing address / citytown]→ New York City | Sony Pictures Classics → New York City (lân cận (không nối về anchor)) | +0.0010 |

Bỏ đồng thời top-8: điểm giảm +0.0092, hạng thay đổi +1.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): -0.0002, +0.0

## that_bai_bi_lam_hai — q_bc8a3779d5d27434fc8fac4d
**Truy vấn:** Find an image of the entity whose artists relation points to Dave Matthews.  
**Hạng ảnh đúng:** CLIP 1 → CLIP + AdaMF 4  
**Hình:** `cases/that_bai_bi_lam_hai_q_bc8a3779d5d27434fc8fac4d.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Roots rock** · điểm đồ thị -0.102 · hạng 4
- Phương thức (E2): attention cấu trúc (KG) 0.35, ảnh 0.20, văn bản 0.45; che từng phương thức: cấu trúc (KG) Δ +0.084 (hạng → 103), ảnh Δ +0.007 (hạng → 4), văn bản Δ +0.071 (hạng → 64)
- Bộ ba train của thực thể: 19 · điểm gốc -0.102 · bản học lại đủ bộ ba +0.033

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Roots rock –[genre / artists]→ John Mellencamp | John Mellencamp → Roots rock (lân cận (không nối về anchor)) | +0.0067 |
| 2 | Roots rock –[genre / artists]→ John Fogerty | John Fogerty → Roots rock (lân cận (không nối về anchor)) | +0.0063 |
| 3 | Roots rock –[genre / parent genre]→ Rock music | Dave Matthews → Rock music → Roots rock (2 bước) | +0.0048 |
| 4 | Roots rock –[genre / artists]→ Jason Mraz | Jason Mraz → Roots rock (lân cận (không nối về anchor)) | +0.0044 |
| 5 | Roots rock –[genre / artists]→ Ry Cooder | Ry Cooder → Roots rock (lân cận (không nối về anchor)) | +0.0027 |
| 6 | Roots rock –[genre / artists]→ Los Lobos | Los Lobos → Roots rock (lân cận (không nối về anchor)) | +0.0025 |
| 7 | Roots rock –[genre / artists]→ Bruce Springsteen | Bruce Springsteen → Roots rock (lân cận (không nối về anchor)) | +0.0020 |
| 8 | Roots rock –[genre / artists]→ Steve Earle | Steve Earle → Roots rock (lân cận (không nối về anchor)) | +0.0015 |

Bỏ đồng thời top-8: điểm giảm +0.0246, hạng thay đổi +9.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0067, +0.5

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Post-rock** · điểm đồ thị -0.067 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.38, ảnh 0.18, văn bản 0.44; che từng phương thức: cấu trúc (KG) Δ +0.117 (hạng → 242), ảnh Δ +0.008 (hạng → 3), văn bản Δ +0.091 (hạng → 98)
- Bộ ba train của thực thể: 17 · điểm gốc -0.067 · bản học lại đủ bộ ba -0.048

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Post-rock –[genre / artists]→ Steven Wilson | Steven Wilson → Post-rock (lân cận (không nối về anchor)) | +0.0086 |
| 2 | Post-rock –[genre / artists]→ The Mars Volta | The Mars Volta → Post-rock (lân cận (không nối về anchor)) | +0.0084 |
| 3 | Post-rock –[genre / artists]→ A-ha | A-ha → Post-rock (lân cận (không nối về anchor)) | +0.0071 |
| 4 | Post-rock –[genre / artists]→ Portishead | Portishead → Post-rock (lân cận (không nối về anchor)) | +0.0030 |
| 5 | Post-rock –[genre / parent genre]→ Experimental rock | Experimental rock → Post-rock (lân cận (không nối về anchor)) | +0.0006 |

Bỏ đồng thời top-8: điểm giảm +0.0848, hạng thay đổi +69.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0034, +1.6

## that_bai_bi_lam_hai — q_4ba6706442617b0d185d2f9c
**Truy vấn:** Find an image of the entity whose ceremony relation points to 60th Primetime Emmy Awards.  
**Hạng ảnh đúng:** CLIP 2 → CLIP + AdaMF 7  
**Hình:** `cases/that_bai_bi_lam_hai_q_4ba6706442617b0d185d2f9c.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special** · điểm đồ thị -0.186 · hạng 7
- Phương thức (E2): attention cấu trúc (KG) 0.26, ảnh 0.13, văn bản 0.60; che từng phương thức: cấu trúc (KG) Δ +0.056 (hạng → 94), ảnh Δ +0.013 (hạng → 15), văn bản Δ +0.257 (hạng → 1000)
- Bộ ba train của thực thể: 26 · điểm gốc -0.186 · bản học lại đủ bộ ba -0.105

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special –[award honor / ceremony]→ 63rd Primetime Emmy Awards | 63rd Primetime Emmy Awards → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (lân cận (không nối về anchor)) | +0.0137 |
| 2 | Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special –[award honor / ceremony]→ 61st Primetime Emmy Awards | 61st Primetime Emmy Awards → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (lân cận (không nối về anchor)) | +0.0133 |
| 3 | Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special –[award honor / ceremony]→ 64th Primetime Creative Arts Emmy Awards | 64th Primetime Creative Arts Emmy Awards → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (lân cận (không nối về anchor)) | +0.0127 |
| 4 | Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special –[award honor / ceremony]→ 58th Primetime Emmy Awards | 58th Primetime Emmy Awards → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (lân cận (không nối về anchor)) | +0.0127 |
| 5 | Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special –[award category / category of]→ Primetime Emmy Award | 60th Primetime Emmy Awards → Primetime Emmy Award → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (2 bước) | +0.0044 |
| 6 | John Adams –[award honor / award]→ Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special | 60th Primetime Emmy Awards → John Adams → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (2 bước) | +0.0004 |
| 7 | Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special –[award nomination / nominated for]→ Alice in Wonderland | Alice in Wonderland → Primetime Emmy Award for Outstanding Costumes for a Miniseries, Movie or a Special (lân cận (không nối về anchor)) | +0.0000 |

Bỏ đồng thời top-8: điểm giảm +0.0966, hạng thay đổi +321.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0060, +7.4

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Primetime Emmy Award for Outstanding Television Movie** · điểm đồ thị -0.130 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.31, ảnh 0.14, văn bản 0.56; che từng phương thức: cấu trúc (KG) Δ +0.101 (hạng → 222), ảnh Δ +0.007 (hạng → 2), văn bản Δ +0.121 (hạng → 349)
- Bộ ba train của thực thể: 32 · điểm gốc -0.130 · bản học lại đủ bộ ba -0.121

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Primetime Emmy Award for Outstanding Television Movie –[award honor / ceremony]→ 59th Primetime Emmy Awards | 59th Primetime Emmy Awards → Primetime Emmy Award for Outstanding Television Movie (lân cận (không nối về anchor)) | +0.0123 |
| 2 | Primetime Emmy Award for Outstanding Television Movie –[award honor / ceremony]→ 61st Primetime Emmy Awards | 61st Primetime Emmy Awards → Primetime Emmy Award for Outstanding Television Movie (lân cận (không nối về anchor)) | +0.0119 |
| 3 | Primetime Emmy Award for Outstanding Television Movie –[award category / category of]→ Primetime Emmy Award | 60th Primetime Emmy Awards → Primetime Emmy Award → Primetime Emmy Award for Outstanding Television Movie (2 bước) | +0.0031 |
| 4 | Barry Levinson –[award nomination / award]→ Primetime Emmy Award for Outstanding Television Movie | Barry Levinson → Primetime Emmy Award for Outstanding Television Movie (lân cận (không nối về anchor)) | +0.0011 |
| 5 | Bury My Heart at Wounded Knee –[award honor / award]→ Primetime Emmy Award for Outstanding Television Movie | Bury My Heart at Wounded Knee → Primetime Emmy Award for Outstanding Television Movie (lân cận (không nối về anchor)) | +0.0004 |
| 6 | Temple Grandin –[award honor / award]→ Primetime Emmy Award for Outstanding Television Movie | Temple Grandin → Primetime Emmy Award for Outstanding Television Movie (lân cận (không nối về anchor)) | +0.0002 |

Bỏ đồng thời top-8: điểm giảm +0.0339, hạng thay đổi +14.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0054, +1.4

## that_bai_bi_lam_hai — q_c5a6595f6b832ef478abab2f
**Truy vấn:** Find an image of the entity whose role relation points to Tubular bell.  
**Hạng ảnh đúng:** CLIP 5 → CLIP + AdaMF 13  
**Hình:** `cases/that_bai_bi_lam_hai_q_c5a6595f6b832ef478abab2f.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Organ** · điểm đồ thị -0.279 · hạng 13
- Phương thức (E2): attention cấu trúc (KG) 0.34, ảnh 0.18, văn bản 0.48; che từng phương thức: cấu trúc (KG) Δ +0.084 (hạng → 244), ảnh Δ -0.005 (hạng → 11), văn bản Δ +0.067 (hạng → 132)
- Bộ ba train của thực thể: 389 · điểm gốc -0.279 · bản học lại đủ bộ ba -0.044

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Organ –[track contribution / role]→ Bell | Tubular bell → Bell → Organ (2 bước) | +0.0010 |
| 2 | Organ –[track contribution / role]→ Tambourine | Tubular bell → Tambourine → Organ (2 bước) | +0.0007 |
| 3 | Organ –[track contribution / role]→ Tenor saxophone | Tubular bell → Tenor saxophone → Organ (2 bước) | +0.0007 |
| 4 | Organ –[track contribution / role]→ Drum | Tubular bell → Drum → Organ (2 bước) | +0.0006 |
| 5 | Organ –[track contribution / role]→ Bass | Tubular bell → Bass → Organ (2 bước) | +0.0006 |
| 6 | Organ –[track contribution / role]→ Percussion | Tubular bell → Percussion → Organ (2 bước) | +0.0005 |
| 7 | Organ –[track contribution / role]→ Sitar | Tubular bell → Sitar → Organ (2 bước) | +0.0004 |
| 8 | Bell –[track contribution / role]→ Organ | Tubular bell → Bell → Organ (2 bước) | +0.0004 |

Bỏ đồng thời top-8: điểm giảm +0.0050, hạng thay đổi +2.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0001, -0.7

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Tin whistle** · điểm đồ thị -0.208 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.39, ảnh 0.16, văn bản 0.45; che từng phương thức: cấu trúc (KG) Δ +0.183 (hạng → 692), ảnh Δ -0.027 (hạng → 1), văn bản Δ +0.028 (hạng → 4)
- Bộ ba train của thực thể: 100 · điểm gốc -0.208 · bản học lại đủ bộ ba -0.039

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Tin whistle –[track contribution / role]→ Melodica | Tubular bell → Melodica → Tin whistle (2 bước) | +0.0017 |
| 2 | Tin whistle –[track contribution / role]→ Piano | Tubular bell → Piano → Tin whistle (2 bước) | +0.0017 |
| 3 | Tin whistle –[track contribution / role]→ Electric guitar | Tubular bell → Electric guitar → Tin whistle (2 bước) | +0.0016 |
| 4 | Tin whistle –[track contribution / role]→ Saxophone | Tubular bell → Saxophone → Tin whistle (2 bước) | +0.0016 |
| 5 | Tin whistle –[track contribution / role]→ Banjo | Tubular bell → Banjo → Tin whistle (2 bước) | +0.0016 |
| 6 | Tin whistle –[track contribution / role]→ Electronic keyboard | Tubular bell → Electronic keyboard → Tin whistle (2 bước) | +0.0015 |
| 7 | Tin whistle –[track contribution / role]→ Tambourine | Tubular bell → Tambourine → Tin whistle (2 bước) | +0.0014 |
| 8 | Tin whistle –[track contribution / role]→ Percussion | Tubular bell → Percussion → Tin whistle (2 bước) | +0.0014 |

Bỏ đồng thời top-8: điểm giảm +0.0149, hạng thay đổi +1.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0008, +0.0

## that_bai_khong_cuu_duoc — q_8985a978b684ff516322ad9b
**Truy vấn:** Find an image of the entity whose film release region relation points to Mexico.  
**Hạng ảnh đúng:** CLIP 920 → CLIP + AdaMF 158  
**Hình:** `cases/that_bai_khong_cuu_duoc_q_8985a978b684ff516322ad9b.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Baraka** · điểm đồ thị -0.182 · hạng 158
- Phương thức (E2): attention cấu trúc (KG) 0.36, ảnh 0.22, văn bản 0.42; che từng phương thức: cấu trúc (KG) Δ +0.084 (hạng → 494), ảnh Δ +0.035 (hạng → 285), văn bản Δ +0.035 (hạng → 283)
- Bộ ba train của thực thể: 29 · điểm gốc -0.182 · bản học lại đủ bộ ba -0.006

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Baraka –[film regional release date / film release region]→ United States of America | Mexico → United States of America → Baraka (2 bước) | +0.0050 |
| 2 | Baraka –[film regional release date / film release region]→ Canada | Mexico → Canada → Baraka (2 bước) | +0.0044 |
| 3 | Baraka –[film regional release date / film release region]→ Hungary | Hungary → Baraka (lân cận (không nối về anchor)) | +0.0043 |
| 4 | Baraka –[film regional release date / film release region]→ Portugal | Portugal → Baraka (lân cận (không nối về anchor)) | +0.0042 |
| 5 | Baraka –[film regional release date / film release region]→ United Kingdom | Mexico → United Kingdom → Baraka (2 bước) | +0.0038 |
| 6 | Baraka –[film regional release date / film release region]→ Republic of Ireland | Republic of Ireland → Baraka (lân cận (không nối về anchor)) | +0.0033 |
| 7 | Baraka –[film regional release date / film release region]→ Turkey | Turkey → Baraka (lân cận (không nối về anchor)) | +0.0029 |
| 8 | Baraka –[film regional release date / film release region]→ France | Mexico → France → Baraka (2 bước) | +0.0028 |

Bỏ đồng thời top-8: điểm giảm +0.0569, hạng thay đổi +204.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0026, +3.6

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Alvin and the Chipmunks: Chipwrecked** · điểm đồ thị -0.109 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.41, ảnh 0.25, văn bản 0.34; che từng phương thức: cấu trúc (KG) Δ +0.177 (hạng → 585), ảnh Δ -0.001 (hạng → 1), văn bản Δ +0.023 (hạng → 13)
- Bộ ba train của thực thể: 61 · điểm gốc -0.109 · bản học lại đủ bộ ba +0.026

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ United States of America | Mexico → United States of America → Alvin and the Chipmunks: Chipwrecked (2 bước) | +0.0014 |
| 2 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ Panama | Panama → Alvin and the Chipmunks: Chipwrecked (lân cận (không nối về anchor)) | +0.0012 |
| 3 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ Netherlands | Mexico → Netherlands → Alvin and the Chipmunks: Chipwrecked (2 bước) | +0.0009 |
| 4 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ Mexico | Mexico → Alvin and the Chipmunks: Chipwrecked (1 bước) | +0.0008 |
| 5 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ Russia | Mexico → Russia → Alvin and the Chipmunks: Chipwrecked (2 bước) | +0.0008 |
| 6 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ Australia | Mexico → Australia → Alvin and the Chipmunks: Chipwrecked (2 bước) | +0.0008 |
| 7 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ Poland | Mexico → Poland → Alvin and the Chipmunks: Chipwrecked (2 bước) | +0.0007 |
| 8 | Alvin and the Chipmunks: Chipwrecked –[film regional release date / film release region]→ United Kingdom | Mexico → United Kingdom → Alvin and the Chipmunks: Chipwrecked (2 bước) | +0.0007 |

Bỏ đồng thời top-8: điểm giảm +0.0087, hạng thay đổi +4.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): -0.0014, +0.0

## that_bai_khong_cuu_duoc — q_7bc9cb9ce8b1f059280a4d39
**Truy vấn:** Find an image of the entity that is related to Minnesota by contains.  
**Hạng ảnh đúng:** CLIP 2 → CLIP + AdaMF 382  
**Hình:** `cases/that_bai_khong_cuu_duoc_q_7bc9cb9ce8b1f059280a4d39.png`


**Ảnh đúng**
- Thực thể quyết định (E1): **Ramsey County** · điểm đồ thị -0.248 · hạng 382
- Phương thức (E2): attention cấu trúc (KG) 0.24, ảnh 0.21, văn bản 0.55; che từng phương thức: cấu trúc (KG) Δ +0.006 (hạng → 430), ảnh Δ +0.020 (hạng → 612), văn bản Δ +0.150 (hạng → 1000)
- Bộ ba train của thực thể: 12 · điểm gốc -0.248 · bản học lại đủ bộ ba -0.185

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | United States of America –[country / second level divisions]→ Ramsey County | Minnesota → United States of America → Ramsey County (2 bước) | +0.0172 |
| 2 | Ramsey County –[location / time zones]→ Central Time Zone | Central Time Zone → Ramsey County (lân cận (không nối về anchor)) | +0.0150 |
| 3 | Ramsey County –[dated money value / currency]→ United States Dollar | Minnesota → United States Dollar → Ramsey County (2 bước) | +0.0144 |
| 4 | Ramsey County –[dated integer / source]→ United States Department of Housing and Urban Development | United States Department of Housing and Urban Development → Ramsey County (lân cận (không nối về anchor)) | +0.0113 |
| 5 | Ramsey County –[adjoining relationship / adjoins]→ Hennepin County | Minnesota → Hennepin County → Ramsey County (2 bước) | +0.0022 |
| 6 | Saint Paul –[administrative division capital relationship / administrative division]→ Ramsey County | Minnesota → Saint Paul → Ramsey County (2 bước) | +0.0015 |
| 7 | Ramsey County –[adjoining relationship / adjoins]→ Dakota County | Minnesota → Dakota County → Ramsey County (2 bước) | +0.0002 |

Bỏ đồng thời top-8: điểm giảm +0.0810, hạng thay đổi +575.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0379, +359.0

**Ảnh xếp hạng 1 (sai)**
- Thực thể quyết định (E1): **Memorial University of Newfoundland** · điểm đồ thị -0.132 · hạng 1
- Phương thức (E2): attention cấu trúc (KG) 0.31, ảnh 0.18, văn bản 0.51; che từng phương thức: cấu trúc (KG) Δ +0.070 (hạng → 195), ảnh Δ +0.024 (hạng → 29), văn bản Δ +0.067 (hạng → 182)
- Bộ ba train của thực thể: 17 · điểm gốc -0.132 · bản học lại đủ bộ ba -0.073

| # | Bộ ba train (bằng chứng) | Đường đi từ anchor | Δ điểm khi bỏ |
|---|---|---|---:|
| 1 | Newfoundland –[location / contains]→ Memorial University of Newfoundland | Newfoundland → Memorial University of Newfoundland (lân cận (không nối về anchor)) | +0.0137 |
| 2 | Newfoundland and Labrador –[location / contains]→ Memorial University of Newfoundland | Newfoundland and Labrador → Memorial University of Newfoundland (lân cận (không nối về anchor)) | +0.0133 |
| 3 | Canada –[location / contains]→ Memorial University of Newfoundland | Canada → Memorial University of Newfoundland (lân cận (không nối về anchor)) | +0.0132 |
| 4 | St. John's –[location / contains]→ Memorial University of Newfoundland | St. John's → Memorial University of Newfoundland (lân cận (không nối về anchor)) | +0.0077 |
| 5 | Memorial University of Newfoundland –[mailing address / state province region]→ Newfoundland and Labrador | Newfoundland and Labrador → Memorial University of Newfoundland (lân cận (không nối về anchor)) | +0.0011 |

Bỏ đồng thời top-8: điểm giảm +0.0709, hạng thay đổi +198.0 · bỏ 8 bộ ba ngẫu nhiên (trung bình 10 lần): +0.0112, +19.2