# Báo cáo giai đoạn xử lý dữ liệu — Nghiệp

## 1. Kết luận nhanh

Trong Phase 2, nhóm data đã xử lý và bàn giao **2 dataset chính**:

1. **DB15K** — dataset gốc của AdaMF, được giữ một bản reproduction và mở rộng thành image-level retrieval; sau đó tạo thêm clean experimental protocol để nhóm model chọn checkpoint/hyperparameter hợp lệ.
2. **FB15K-237-IMG** — dataset thứ hai được adapter sang định dạng AdaMF và bổ sung toàn bộ pipeline ảnh, feature, mask và retrieval bridge.

Không tính vào con số trên:

- **MKG-W** chỉ đạt mức candidate KGC-only, không được promote vì thiếu raw image/entity-image mapping có thể kiểm chứng và thiếu sidecar provenance cho tensor.
- **Visual Genome** thuộc nhánh dữ liệu scene graph khác: triple chủ yếu mô tả object/relationship cục bộ trong ảnh, không có cùng entity-level KG contract với AdaMF. Vì vậy không được chọn làm dataset 2 của Phase 2.

| Dataset | Kết quả data-side | Mục đích chính |
|---|---|---|
| DB15K | Reproduction + canonical image retrieval + clean protocol | Tái lập AdaMF và huấn luyện/đánh giá retrieval với validation sạch |
| FB15K-237-IMG | AdaMF adapter + image/text features + retrieval bridge | Dataset thứ hai để kiểm tra khả năng tổng quát hóa |

---

## 2. Goal dữ liệu của dự án

Dataset đầu ra không chỉ cần chạy được KGC. Nó phải hỗ trợ đồng thời:

- official transductive link prediction cho cả head `(?, r, t)` và tail `(h, r, ?)`;
- image-level retrieval, trong đó candidate là **ảnh**, không phải entity;
- mọi ảnh hợp lệ của target entity trong đúng split đều là positive;
- điểm KG của entity được broadcast xuống ảnh, còn CLIP dùng để phân biệt các ảnh của cùng entity;
- chọn normalization, hệ số fusion `alpha`, checkpoint và hyperparameter chỉ trên validation;
- có mapping rõ ràng giữa entity, image, feature row và modality mask;
- ngăn exact-image duplicate đi qua các retrieval split;
- giữ checksum, manifest và protocol version để nhóm model tái lập đúng dữ liệu;
- không trộn kết quả reproduction với kết quả clean/adapter protocol.

---

## 3. Dataset 1 — DB15K

### 3.1. Tính chất gốc và khác biệt so với goal

DB15K upstream là benchmark multimodal KGC native của AdaMF:

| Thuộc tính gốc | Giá trị |
|---|---:|
| Entity | 12.842 |
| Relation | 279 |
| Train | 79.222 triples |
| Official validation | 9.904 rows |
| Official test | 9.902 triples |
| Entity visual tensor | `[12842, 4096]` |
| Entity textual tensor | `[12842, 768]` |

Các điểm chưa phù hợp với goal:

1. Upstream chủ yếu cung cấp **entity-level feature** cho KGC, chưa cung cấp image-level CLIP tensor, image index, retrieval split, query/relevance hay candidate pool.
2. Official validation không phù hợp để chọn model mới:
   - dư 2.423 duplicate rows;
   - 6.584 unique triples trùng train;
   - 824 unique triples trùng test;
   - chỉ còn 73 unique triples sau khi loại overlap với train và test.
3. Official test có 84 cold-start triples và 131 reverse-same-relation triples cần báo riêng khi đánh giá.
4. Tensor upstream không có entity-key sidecar độc lập; row alignment chỉ đạt `PASS_WITH_PROVENANCE_LIMITATION`.
5. Raw image/web-image provenance và quyền redistribution chưa đủ để dùng như một benchmark ảnh công khai.

Vì hai mục tiêu khác nhau, DB15K được giữ thành **hai nhánh song song**:

- official reproduction: giữ nguyên byte để tái lập upstream;
- clean experimental protocol: dùng cho model selection và thí nghiệm mới.

### 3.2. Kỹ thuật đã áp dụng

#### A. Reproduction và integrity audit

- Pin AdaMF source commit và toàn bộ input bằng SHA-256.
- Kiểm tra OpenKE mappings, ID liên tục, row counts, duplicate và overlap.
- Kiểm tra tensor shape/dtype/finite/zero-row và loader contract.
- Giữ nguyên official train/valid/test và type constraints; không “làm sạch” bản reproduction.
- Chạy AdaMF loader/head-tail/optimizer smoke để xác nhận compatibility, không tuyên bố paper-metric reproduction.

#### B. Xây canonical image-level retrieval

- Parse entity-image mapping và thu thập ảnh vào namespace riêng.
- Decode audit, checksum, pixel-level exact duplicate detection và entity-image integrity check.
- Canonicalize ảnh RGB; loại artifact không decode được/không hợp lệ.
- Tạo image-disjoint split bằng exact-duplicate cluster, seed 2026; không claim entity-disjoint.
- Extract CLIP image feature 512 chiều, float32, L2-normalized; tạo `image_feature_index` để khóa row alignment.
- Tạo direct query và indirect query cho cả head/tail.
- Tạo multi-positive relevance và candidate pool theo cùng retrieval split.
- Freeze ranking contract:

```text
S_final = alpha * normalize(S_CLIP) + (1 - alpha) * normalize(S_KG)
```

#### C. Tạo clean experimental protocol

- Chỉ chia lại **official train**, không lấy dữ liệu từ official valid/test.
- Chia target 90/10 với seed 2027 và SHA-256 deterministic priority.
- Group `(min(h,t), max(h,t), r)` để cạnh đảo cùng relation không bị tách qua train/valid.
- Relation-stratified assignment kèm ràng buộc giữ entity/relation coverage trong clean train.
- Giữ official test byte-identical.
- Sinh test views `seen-only` và `cold-start`.
- Regenerate `type_constrain.txt` từ clean train only.
- Regenerate clean retrieval query bridge; chuẩn hóa relation label bằng strip URI delimiter, URL decode, split camelCase và normalize khoảng trắng.
- Chạy lại loader/model smoke và freeze model-facing training interface.

### 3.3. Dataset sau xử lý

#### Official reproduction

- 12.842 entity, 279 relation.
- Train/valid/test giữ nguyên 79.222/9.904/9.902.
- Visual `[12842,4096]`, textual `[12842,768]`.
- Dùng khi cần tái lập AdaMF/upstream; không dùng official validation để chọn model mới.

#### Canonical retrieval `db15k_v1`

| Thành phần | Kết quả |
|---|---:|
| Canonical images | 25.925 |
| CLIP image feature | `[25925,512]` |
| Query gồm test | 155.679 |
| Relevance rows | 319.469 |
| Query types | direct + indirect head/tail |

#### Clean protocol `db15k_clean_from_train_v1`

| Thành phần | Kết quả |
|---|---:|
| Clean train | 71.300 |
| Clean validation | 7.922 |
| Official test, byte-identical | 9.902 |
| Test seen-only | 9.818 |
| Test cold-start | 84 |
| Entity/relation được phủ trong clean train | 12.785 / 273 |
| Clean retrieval queries | 141.488 |
| Query train/val/test | 130.988 / 5.023 / 5.477 |
| Relevance rows | 290.087 |
| Candidate images | 25.925 |

Clean split đạt zero exact overlap/duplicate giữa train-valid-test và zero reverse-same-relation group chéo clean train/valid. Official validation bị loại hoàn toàn khỏi clean protocol. Final data/loader replay đạt 34/34 checks; việc replay được thực hiện theo quyền nhóm model giao cho Nghiệp và được ghi nhận là delegated replay.

### 3.4. Tính chất cần thiết cho nhóm model

- Với reproduction result, dùng `processed/db15k/reproduction/` và ghi rõ official validation có known overlap/duplicate.
- Với thí nghiệm mới, bắt buộc dùng `processed/db15k/experimental_clean_v1/reproduction/`.
- Train lại mọi model từ đầu trên cùng clean train; không dùng checkpoint từ official protocol để so sánh trực tiếp.
- Early stopping, checkpoint, normalization và hyperparameter chỉ chọn trên clean validation.
- Official test chỉ được mở sau khi config đã freeze.
- Báo metric riêng full test, seen-only, cold-start, head, tail và combined.
- Retrieval phải dùng query/relevance/candidate của `db15k_clean_v1`; không trộn với query của canonical `db15k_v1`.
- Feature row vẫn có provenance limitation; không được tự ý reorder entity hoặc tensor row.
- Ảnh chỉ dùng trong phạm vi coursework; không redistribute raw images.

---

## 4. Dataset 2 — FB15K-237-IMG

### 4.1. Tính chất gốc và khác biệt so với goal

Nguồn ban đầu là sự kết hợp của:

- official FB15K-237 symbolic KG từ MKGformer;
- entity text;
- MMKB image index và external image archive.

KG gốc có:

| Thuộc tính | Giá trị |
|---|---:|
| Entity | 14.541 |
| Relation | 237 |
| Train | 272.115 |
| Validation | 17.535 |
| Test | 20.466 |

Khác biệt với goal:

1. Đây không phải native benchmark của AdaMF-MAT, nên chưa có OpenKE/AdaMF adapter được validate.
2. Ảnh nằm trong external archive; filename/mapping kỹ thuật chưa đủ để chứng minh đúng entity.
3. Có ảnh nhỏ, duplicate trong cùng entity và exact duplicate liên kết nhiều entity.
4. Chưa có image-disjoint retrieval split, CLIP tensor, entity pooling, modality mask hay query bridge.
5. Không phải entity nào cũng có ảnh; model cần missing-modality policy rõ ràng.
6. Quyền sử dụng ảnh chỉ đạt conditional coursework use; không được redistribute raw images.

### 4.2. Kỹ thuật đã áp dụng

#### A. Source, archive và mapping audit

- Pin MKGformer/MMKB sources và archive bằng checksum.
- Inventory 242.860 archive members, kiểm duplicate name/path và safe extraction.
- Decode thành công 146.019/146.019 benchmark source files.
- Loại 703 ảnh nhỏ hơn 64×64.
- Join entity-image bằng Freebase key thay vì dựa vào line order.
- Kiểm coverage, orphan, missing entity, image hash và pixel hash.

#### B. Duplicate và semantic policy

- Phát hiện 1.854 exact-duplicate clusters, trong đó 1.577 cluster chéo entity.
- Áp dụng conservative policy: loại toàn bộ 1.577 cross-entity clusters.
- Collapse 293 same-entity duplicate rows thành canonical image.
- Loại 5.209 mapping rows có rủi ro; còn 139.814 canonical image/mapping rows.
- Semantic pilot 200 ảnh theo source group; policy giữ ảnh được human review.
- Ghi rõ limitation: 200 nhãn `yes` là blanket acceptance dựa trên seeded 10-image visual spot-check, không phải 200 ảnh được xem độc lập.

#### C. AdaMF adapter và retrieval split

- Chuyển official symbolic triples sang OpenKE order `h t r`.
- Sinh `entity2id`, `relation2id`, train/valid/test và train-derived type constraints.
- Xác minh 6 adapter files byte-identical với immutable preview.
- Join text theo entity key, đạt 14.541/14.541 coverage.
- Chia ảnh 80/10/10 gần đúng bằng SHA-256 của exact-duplicate cluster, seed 2026.
- Assert zero duplicate cluster chéo retrieval split.

#### D. Feature và missing modality

- Extract CLIP image feature `[139814,512]`, float32, finite, L2-normalized.
- Mean-pool các ảnh theo entity rồi re-normalize thành visual entity tensor `[14541,512]`.
- Tạo textual entity tensor `[14541,512]`.
- Với 280 entity thiếu visual: zero-fill visual row và tạo explicit modality mask.
- Khóa row alignment bằng `image_feature_index`, `entity_feature_index` và SHA-256.
- Chạy mask-aware one-batch smoke: finite score/gradient, parameter update dương, missing-visual attention bằng 0.

#### E. Retrieval bridge và handoff

- Sinh direct, indirect-head và indirect-tail query.
- Mọi ảnh hợp lệ của target entity trong cùng split là positive.
- Candidate pool gồm toàn bộ canonical images của cùng split.
- Test query chỉ được sinh sau khi protocol/template được review và freeze.
- Final validation kiểm counts, schema, checksum, adapter, feature, query/relevance/candidate consistency.
- Promotion chỉ tạo manifest/reference; không copy lại raw images hoặc tensor lớn.

### 4.3. Dataset sau xử lý

| Thành phần | Kết quả |
|---|---:|
| KG train/valid/test | 272.115 / 17.535 / 20.466 |
| Canonical images | 139.814 |
| Entity có ảnh | 14.261 / 14.541 |
| Entity thiếu ảnh | 280 |
| Visual coverage | 98,074410% |
| Image split train/val/test | 111.906 / 13.971 / 13.937 |
| Image feature | `[139814,512]` |
| Entity visual/textual | `[14541,512]` / `[14541,512]` |
| Retrieval queries | 604.084 |
| Query train/val/test | 541.840 / 29.088 / 33.156 |
| Relevance rows | 4.392.803 |
| Candidate rows | 139.814 |

Dataset đạt trạng thái data-side:

```text
FB15K237_IMAGE_RETRIEVAL_READY_WITH_COURSEWORK_LICENSE_LIMITATION
```

Final validation đạt 15/15 checks, `failures=[]`. Protocol là transductive KG + image-disjoint retrieval, không phải entity-disjoint retrieval và không phải AdaMF paper reproduction.

### 4.4. Tính chất cần thiết cho nhóm model

- Dùng đúng 14.541-row entity index cho KG, visual, textual và modality mask.
- Luôn áp modality mask trước fusion; 280 zero visual rows không được xem như ảnh thật.
- Không tự reorder tensor hoặc mapping rows; phải join bằng index sidecar/checksum.
- Chọn `alpha`, normalization, checkpoint và hyperparameter trên validation; không tune trên test.
- Log head, tail, combined và direct retrieval riêng.
- Retrieval candidate/positive phải được giới hạn theo frozen image split.
- Không được suy từ loader smoke rằng đã tái lập metric AdaMF paper.
- CLIP model revision trả `resolved_commit=null`; exact feature được tái lập bằng frozen artifact hash, không claim commit-pinned model reproduction.
- Raw images chỉ dùng trong Drive/coursework và không được redistribute.

---

## 5. Các tính chất chung quan trọng nhất cho huấn luyện

| Tính chất | Vì sao cần cho nhóm model |
|---|---|
| Protocol ID và namespace riêng | Ngăn trộn official reproduction, clean DB15K và FB15K-237 adapter trong cùng bảng kết quả |
| Validation độc lập, không overlap | Cho phép early stopping và chọn hyperparameter mà không rò rỉ test |
| Official test được freeze | Giữ đánh giá cuối công bằng và có thể so sánh |
| Entity/feature row alignment | Tránh entity nhận nhầm visual/textual feature |
| Explicit modality mask | Tránh zero-fill của entity thiếu ảnh bị model coi là tín hiệu thật |
| Image-disjoint theo duplicate cluster | Ngăn cùng nội dung ảnh xuất hiện ở cả train và test |
| Multi-positive relevance | Đánh giá đúng khi một entity có nhiều ảnh hợp lệ |
| Candidate pool theo split | Ngăn retrieval nhìn thấy ảnh ngoài split và gây leakage |
| Head/tail tách riêng | Phát hiện model mạnh một chiều nhưng yếu chiều còn lại |
| Seen-only/cold-start tách riêng | Không che giấu lỗi do entity/relation chưa thấy |
| Checksum, manifest, sidecar | Cho phép replay đúng artifact thay vì chỉ dựa vào tên file |
| License/provenance limitation | Giới hạn đúng phạm vi sử dụng và tránh tuyên bố quá mức |

## 6. Artifact bàn giao chính

### DB15K

```text
dataset/adamf/processed/db15k/reproduction/
dataset/adamf/processed/db15k/experimental_clean_v1/
dataset/adamf/db15k_image_retrieval/runs/db15k_v1/
dataset/adamf/db15k_image_retrieval/runs/db15k_clean_v1/
dataset/adamf/manifests/db15k_clean_protocol_v1_manifest.json
dataset/adamf/processed/db15k/experimental_clean_v1/training_interface.json
dataset/adamf/DB15K_CLEAN_MODEL_HANDOFF.md
```

### FB15K-237-IMG

```text
dataset/adamf/fb15k237_image_retrieval/runs/fb15k237_v1/
dataset/adamf/processed/fb15k237_img/retrieval_bridge/manifest.json
dataset/adamf/manifests/fb15k237_img_v1_manifest.json
dataset/adamf/reports/fb15k237_img_v1_final_validation.json
dataset/adamf/FB15K237_DATA_HANDOFF.md
```

## 7. Kết luận bàn giao

Phần data đã chuyển hai nguồn KG đa phương thức có chất lượng và cấu trúc khác nhau thành hai protocol có thể được nhóm model tiêu thụ theo cùng nguyên tắc: **KG triples + aligned multimodal features + image-level candidates + frozen query/relevance contract**.

DB15K cung cấp cả nhánh reproduction và nhánh clean để model selection hợp lệ. FB15K-237-IMG cung cấp phép thử dataset thứ hai với adapter, image-disjoint retrieval và missing-modality mask. Phần còn lại thuộc nhóm model là train lại mô hình, chọn checkpoint/alpha trên validation và báo metric theo đúng protocol; nhóm data không tuyên bố convergence hoặc paper-level performance.
