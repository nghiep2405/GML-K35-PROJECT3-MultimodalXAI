# Kịch bản thuyết trình Project 03 — Truy hồi đa phương thức có giải thích bằng đồ thị

Tài liệu bám theo `slide/output/final_presentation.pdf` sau đợt sửa nội dung ngày 21/11/2026: **15 slide chính và 8 slide dự phòng, tổng cộng 23 trang PDF**. Các trang dự phòng vẫn hiển thị 15/15 ở chân trang; tài liệu này dùng số trang PDF 16–23 và ký hiệu B1–B8 để dễ tra cứu.

Mỗi slide gồm lời thuyết trình có thể nói trực tiếp, kiến thức để hiểu và bảo vệ nội dung, cùng đúng ba câu hỏi dự kiến. Phần kiến thức và hỏi đáp dùng để chuẩn bị, không đọc toàn bộ khi trình bày. Các slide dự phòng chỉ mở khi cần trả lời câu hỏi. Kịch bản demo là thao tác dự kiến; chỉ tuyên bố một bước đã chạy thành công khi có kết quả thực tế trên màn hình.

## Slide 1 — Giới thiệu đề tài

**Tiêu đề trên slide:** Multimodal Retrieval with Graph-based Explainability.

### Lời thuyết trình

Kính chào thầy và các bạn. Nhóm 02 xin trình bày Project 03 về truy hồi đa phương thức có giải thích bằng đồ thị. Bài toán chính của nhóm là nhận một câu truy vấn văn bản và trả về những hình ảnh phù hợp.

Nhóm tập trung vào hai câu hỏi. Thứ nhất, biểu diễn đồ thị có giúp cải thiện thứ hạng so với CLIP hay không, đặc biệt khi đáp án phải được xác định qua quan hệ với một thực thể khác? Thứ hai, khi kết quả thay đổi, nhóm có thể xác định những bằng chứng đồ thị ảnh hưởng đến điểm xếp hạng hay không?

Phần trình bày đi từ scene graph trong từng ảnh, sang đồ thị tri thức đa phương thức, rồi đến đánh giá giải thích và thực nghiệm trên dữ liệu di sản Việt Nam do nhóm xây dựng.

### Giải thích kiến thức

- **Truy hồi đa phương thức:** truy vấn và đối tượng tìm kiếm thuộc các dạng dữ liệu khác nhau; ở đây là văn bản tìm ảnh.
- **Đồ thị:** các đỉnh biểu diễn đối tượng hoặc thực thể, cạnh biểu diễn quan hệ. Đồ thị có thể cục bộ trong một ảnh hoặc liên kết tri thức giữa nhiều thực thể.
- **XAI:** phân tích bằng chứng liên quan đến quyết định của mô hình. Trong đồ án, giải thích được kiểm tra bằng tác động lên điểm hoặc thứ hạng, không chỉ bằng hình vẽ đường đi.
- Phân biệt ba mục tiêu: truy hồi tốt hơn, xác định đóng góp của cấu trúc, và giải thích đáng tin cậy. Đạt mục tiêu đầu không tự động bảo đảm hai mục tiêu sau.

### Câu hỏi dự kiến và trả lời ngắn

1. **Đầu vào và đầu ra cụ thể là gì?** Đầu vào là câu truy vấn; đầu ra là danh sách ảnh được xếp hạng, kèm bằng chứng đồ thị ở nhánh giải thích.
2. **Đóng góp chính của nhóm nằm ở đâu?** Thích nghi CLIP và mô hình đồ thị cho truy hồi ảnh, xây đối chứng kiểm tra vai trò của cạnh, đánh giá giải thích bằng can thiệp và xây ViHeritage-MMKG.
3. **Nhóm có đề xuất mô hình nền tảng hoàn toàn mới không?** Không. Nhóm sử dụng các mô hình nền có sẵn và tập trung vào triển khai, thích nghi, kiểm chứng thực nghiệm và phân tích giới hạn.

## Slide 2 — Vì sao cần xem xét thông tin quan hệ?

**Tiêu đề trên slide:** CLIP Retrieval Can Struggle with Relational Queries.

### Lời thuyết trình

CLIP đưa văn bản và hình ảnh vào một không gian biểu diễn chung, rồi dùng độ tương đồng cosine để xếp hạng. Đây là đường cơ sở mạnh và là bước đầu của các pipeline trong đồ án.

Tuy nhiên, có những tình huống chỉ nhận ra các đối tượng chưa đủ để phân biệt ảnh. Ví dụ minh họa trên slide là một đứa trẻ đang chơi bóng: ảnh có trẻ em và quả bóng chưa chắc thể hiện đúng hành động được hỏi. Một số truy vấn khác còn đòi hỏi kiến thức ngoài ảnh, chẳng hạn tìm hình của người có quan hệ nhất định với một nhân vật.

Vì vậy, nhóm khảo sát thông tin quan hệ từ đồ thị. Cần lưu ý hình ở đây chỉ minh họa động cơ nghiên cứu; nhóm không kết luận CLIP hoàn toàn không hiểu quan hệ. Ngoài độ chính xác, nhóm cũng muốn biết bằng chứng nào tác động đến kết quả.

### Giải thích kiến thức

- **Embedding:** vector biểu diễn dữ liệu; những mẫu phù hợp về ngữ nghĩa được kỳ vọng nằm gần nhau theo độ đo đã chọn.
- **Cosine similarity:** $\cos(x,y)=x^\top y/(\|x\|\|y\|)$. Với vector đã chuẩn hóa L2, cosine chính là tích vô hướng.
- CLIP có thể học tín hiệu quan hệ từ tiền huấn luyện, nhưng pipeline cosine không thực hiện suy luận đồ thị tường minh.
- **Đối tượng và quan hệ:** cùng tập đối tượng không có nghĩa cùng ngữ nghĩa. “Người đứng cạnh xe” khác “người đang lái xe”.
- Điểm tương đồng cho biết mức phù hợp theo mô hình, không tự chỉ ra quan hệ nào là bằng chứng quyết định.

### Câu hỏi dự kiến và trả lời ngắn

1. **CLIP có hoàn toàn không hiểu quan hệ không?** Không. Nhóm chỉ cho rằng có truy vấn quan hệ khó đối với đường cơ sở và cần kiểm tra lợi ích bổ sung của đồ thị.
2. **Ví dụ trẻ em chơi bóng có phải kết quả của nhóm không?** Không. Đây là ví dụ minh họa bài toán, không phải case thực nghiệm.
3. **Vì sao không fine-tune CLIP luôn?** Nhóm giữ CLIP cố định để có đường cơ sở nhất quán và tập trung khảo sát nhánh đồ thị; chưa kết luận fine-tune CLIP là phương án kém hơn.

## Slide 3 — Ba cấp độ của đồ án

**Tiêu đề trên slide:** Three Levels: Retrieval → Knowledge → Explanation.

### Lời thuyết trình

Nhóm tổ chức công việc thành ba cấp độ. Cấp độ 1 dùng Visual Genome: mỗi ảnh có một scene graph và GAT học biểu diễn đồ thị để xếp hạng lại kết quả CLIP.

Cấp độ 2 mở rộng sang DB15K và FB15K-237-IMG. Ở đây, đồ thị liên kết các thực thể bằng quan hệ tri thức; nhóm dùng AdaMF để dung hợp cấu trúc, văn bản và hình ảnh, rồi đánh giá riêng truy vấn trực tiếp và gián tiếp.

Cấp độ 3 kiểm tra giải thích bằng các phép can thiệp và đưa phương pháp sang ViHeritage-MMKG, bộ dữ liệu di sản tiếng Việt do nhóm xây dựng. XAI không chỉ xuất hiện ở bộ dữ liệu tự xây mà được đánh giá trên cả các cấp độ trước. Nhánh truy hồi chính xuyên suốt là văn bản tìm ảnh.

### Giải thích kiến thức

- **Scene graph:** đối tượng và quan hệ trong một ảnh, ví dụ người–cầm–microphone.
- **MMKG:** đồ thị tri thức đa phương thức; thực thể có thể gắn tên, mô tả, ảnh và các quan hệ ngoài nội dung nhìn thấy trực tiếp.
- **Truy vấn trực tiếp:** nêu thực thể cần tìm. **Truy vấn gián tiếp:** cho thực thể neo và quan hệ để tìm thực thể còn thiếu.
- Ba cấp độ không phải ba lớp của một mạng duy nhất; đó là các nhánh thực nghiệm dùng chung ý tưởng truy hồi kết hợp điểm đồ thị.
- Ba bộ công khai trong mạch slide chính là Visual Genome, DB15K và FB15K-237-IMG; bộ tự xây là ViHeritage.

### Câu hỏi dự kiến và trả lời ngắn

1. **Cấp độ 2 khác cấp độ 1 ở đâu?** Cấp độ 1 khai thác đồ thị trong từng ảnh; cấp độ 2 dùng quan hệ tri thức giữa thực thể, có thể nằm ngoài nội dung trực tiếp của ảnh.
2. **Cấp độ 3 có chỉ là vẽ đồ thị không?** Không. Nhóm đo tác động của bằng chứng bằng can thiệp, so với đối chứng và phân tích cả trường hợp sai.
3. **ViHeritage có thay thế hai benchmark công khai không?** Không. Nó bổ sung phép thử trên miền di sản tiếng Việt, với quy mô và giao thức riêng.

## Slide 4 — Pipeline truy hồi chung

**Tiêu đề trên slide:** One Shared Protocol, Three Graph Settings.

### Lời thuyết trình

Pipeline bắt đầu bằng việc mã hóa câu truy vấn và tính tương đồng với ảnh bằng CLIP. Visual Genome giữ top 50 ảnh, hai benchmark KG giữ top 1.000; ViHeritage đánh giá trên toàn bộ ảnh test của bộ nhỏ này.

Sau đó, GAT hoặc AdaMF cung cấp điểm đồ thị cho các ứng viên. Hai dãy điểm được chuẩn hóa riêng theo truy vấn trước khi kết hợp. Hệ số alpha điều chỉnh trọng số của điểm CLIP; phần còn lại là trọng số điểm đồ thị. Cuối cùng, hệ thống xếp hạng lại và có thể phân tích bằng chứng cho ảnh trả về.

Mọi lựa chọn checkpoint, alpha và cấu hình được thực hiện trên validation. Với DB15K và ViHeritage, truy vấn trực tiếp giữ thứ hạng CLIP theo thiết kế; không phải mọi truy vấn đều đi qua cùng một phép xếp hạng lại có đồ thị.

### Giải thích kiến thức

- Công thức dung hợp: $S(q,i)=\alpha z(S_{CLIP}(q,i))+(1-\alpha)z(S_{graph}(q,i))$.
- **Z-score:** lấy điểm trừ trung bình rồi chia độ lệch chuẩn của dãy điểm ứng viên. Mục đích là giảm khác biệt thang đo, không biến điểm thành xác suất đúng.
- $\alpha=1$: chỉ dùng điểm CLIP để xếp hạng. $\alpha=0$: chỉ dùng điểm đồ thị trong bước rerank, nhưng CLIP có thể vẫn quyết định tập ứng viên.
- **Candidate generation** tạo tập ảnh để xét; **reranking** thay đổi thứ tự bên trong tập đó.
- “Test một lần” được hiểu theo cấu hình đã khóa, không dùng test để điều chỉnh tham số sau khi thấy kết quả.

### Câu hỏi dự kiến và trả lời ngắn

1. **Vì sao chuẩn hóa trước khi cộng điểm?** Điểm CLIP và đồ thị có thang đo khác nhau; nếu cộng trực tiếp, thang đo lớn có thể lấn át trọng số alpha.
2. **Alpha bằng 0 có loại bỏ CLIP hoàn toàn không?** Không nhất thiết. CLIP vẫn tạo top ứng viên trước khi điểm đồ thị xếp hạng lại.
3. **Vì sao chỉ chọn alpha trên validation?** Để test giữ vai trò đánh giá cuối và không bị dùng gián tiếp để tối ưu hệ thống.

## Slide 5 — Scene graph và GAT trên Visual Genome

**Tiêu đề trên slide:** Scene Graph + GAT Rerank CLIP’s Top-50.

### Lời thuyết trình

Với mỗi ảnh Visual Genome, nhóm xây một đồ thị độc lập gồm nút ảnh và các nút đối tượng. Nút ảnh dùng đặc trưng CLIP của ảnh, nút đối tượng dùng đặc trưng văn bản của tên đối tượng; các đặc trưng đầu vào có 512 chiều. Cạnh chứa thông tin quan hệ, có thêm cạnh ngược và cờ phân biệt chiều ngược.

GAT hai lớp truyền thông tin qua đồ thị để tạo biểu diễn ảnh 128 chiều; truy vấn cũng được chiếu từ 512 xuống 128 chiều để chấm điểm tương thích. Điểm này được kết hợp với CLIP, với alpha bằng 0,6 chọn trên validation.

Nhóm huấn luyện bằng AdamW trong 100 epoch, batch tám truy vấn và 50 hard negatives cho mỗi truy vấn. Điều quan trọng là GAT chỉ rerank top 50, nên không khắc phục được mọi lỗi của bước tìm ứng viên.

### Giải thích kiến thức

- **GAT:** học trọng số chú ý khi tổng hợp thông tin từ các nút lân cận; nhiều head cho phép học nhiều cách kết hợp tín hiệu.
- Bốn head, mỗi head 32 chiều, tương ứng biểu diễn 128 chiều theo kiến trúc được dùng; không nhầm với mô hình Flickr30k có cấu hình khác.
- **Cạnh ngược:** giúp tín hiệu đi theo chiều cần thiết trong message passing; không có nghĩa quan hệ tri thức gốc tự trở thành đối xứng.
- **Hard negatives:** ảnh sai nhưng có điểm CLIP cao, buộc mô hình học phân biệt những ứng viên khó.
- Điểm đồ thị dùng biểu diễn truy vấn và nút ảnh. Trọng số attention không tự động là độ quan trọng theo từng truy vấn, vì bộ mã hóa đồ thị không nhận truy vấn như đầu vào trực tiếp.

### Câu hỏi dự kiến và trả lời ngắn

1. **Đặc trưng CLIP có được cập nhật không?** Không. CLIP được đóng băng; GAT và các thành phần chiếu liên quan được huấn luyện.
2. **Vì sao cần cạnh ngược?** Để truyền thông tin theo cả hai chiều trong biểu diễn tính toán, đồng thời vẫn phân biệt chiều bằng metadata cạnh.
3. **Ảnh đúng đứng hạng 80 của CLIP có được đưa lên hạng 1 không?** Không trong giao thức top 50 này, vì ảnh đó không nằm trong tập rerank.

## Slide 6 — Kết quả truy hồi Visual Genome

**Tiêu đề trên slide:** Graph Reranking Improves Retrieval on Visual Genome.

### Lời thuyết trình

Biểu đồ so sánh CLIP thuần với CLIP kết hợp GAT trên cùng bản chia dữ liệu. MRR tăng từ khoảng 0,0611 lên 0,0718, với độ lệch chuẩn khoảng 0,0002 qua ba seed. Theo số liệu chưa làm tròn trong báo cáo, mức tăng tương đối xấp xỉ 17,4%. Recall@1 tăng từ 0,0295 lên 0,0363.

Điều này cho thấy pipeline có học đồ thị cải thiện thứ hạng trong giao thức đang xét. Độ lệch chuẩn nhỏ cho thấy ba lượt chạy cho kết quả gần nhau, nhưng chưa chứng minh mô hình tổng quát hóa tốt sang mọi miền ảnh.

Kết quả này cũng chưa trả lời toàn bộ cải thiện đến từ đâu. Để tách ảnh hưởng của biểu diễn có học với các cạnh quan hệ đúng, nhóm thực hiện các đối chứng ở slide tiếp theo.

### Giải thích kiến thức

- **MRR:** trung bình nghịch đảo thứ hạng ảnh liên quan tốt nhất. Hạng 1 đóng góp 1; hạng 2 đóng góp 0,5; hạng 10 đóng góp 0,1.
- **Recall@1:** tỷ lệ truy vấn có ít nhất một ảnh đúng ở vị trí đầu tiên; không phải độ chính xác phát hiện đối tượng.
- **Tăng tương đối:** $(MRR_{mới}-MRR_{gốc})/MRR_{gốc}$. Khác với tăng tuyệt đối khoảng 0,0107 MRR.
- Độ lệch chuẩn giữa seed không phải khoảng tin cậy của mọi kết luận; đặc biệt không thay thế đánh giá chênh lệch giữa hai cấu hình ablation.
- MRR thấp còn gắn với truy vấn mô tả vùng trong khi ứng viên là cả ảnh và không gian ứng viên lớn.

### Câu hỏi dự kiến và trả lời ngắn

1. **17,4% có nghĩa MRR tăng thêm 0,174 không?** Không. Đó là mức tăng tương đối; mức tăng tuyệt đối khoảng 0,0107.
2. **Độ lệch chuẩn nhỏ có chứng minh vượt trội có ý nghĩa thống kê không?** Không tự nó. Nó mô tả biến thiên của mô hình qua seed; cần phép so sánh phù hợp cho chênh lệch.
3. **Vì sao chưa thể nói cạnh quan hệ tạo toàn bộ mức tăng?** GAT và phép chiếu thêm các tham số học được; phải dùng đối chứng bỏ hoặc xáo cạnh để kiểm tra.

## Slide 7 — Ablation Visual Genome

**Tiêu đề trên slide:** Most of the Gain Survives Without Real Edges.

### Lời thuyết trình

Đây là kết quả giúp nhóm diễn giải thận trọng hơn. Khi bỏ cạnh nhưng giữ thành phần học được, mô hình vẫn giữ khoảng 77% mức tăng so với CLIP. Khi xáo cạnh, khoảng 92% mức tăng còn tồn tại. Full Graph tốt hơn Random-edges khoảng 0,0009 MRR trong lượt chạy này.

Như vậy, phần lớn cải thiện không đòi hỏi các liên kết quan hệ đúng, mà có thể đến từ biến đổi đặc trưng và cơ chế học có giám sát. XAI cung cấp quan sát tương ứng: khoảng 75% phần tử giải thích là cạnh contains, chủ yếu đưa thông tin về sự hiện diện của đối tượng vào nút ảnh.

Case “peace sign” đưa ảnh đúng từ hạng 15 lên hạng 1, nhưng sau khi bỏ tập giải thích ảnh tụt xuống hạng 4. Nhóm xem đây là bằng chứng trong giao thức hiện tại; ablation mới có một seed.

### Giải thích kiến thức

- **No-edge:** bỏ liên kết message passing nhưng giữ đặc trưng và kiến trúc học được; khác với việc quay về CLIP thuần.
- **Random-edges:** hoán vị đích cạnh, giữ những đặc tính cấu trúc được mô tả trong giao thức nhưng phá liên kết đúng.
- Tỷ lệ lợi ích còn giữ: $(MRR_{đối\ chứng}-MRR_{CLIP})/(MRR_{Full}-MRR_{CLIP})$.
- “77% còn giữ” không có nghĩa 77% hiệu năng tổng, và không chứng minh chính xác 77% nguyên nhân đến từ lớp chiếu.
- Cạnh contains vẫn là một loại cạnh đồ thị, nhưng biểu diễn sự có mặt của đối tượng; cần phân biệt với quan hệ giữa hai đối tượng như holding hoặc wearing.

### Câu hỏi dự kiến và trả lời ngắn

1. **Vậy GAT có vô ích không?** Không. Pipeline cải thiện truy hồi; ablation chỉ cho thấy không được quy toàn bộ cải thiện cho ngữ nghĩa cạnh.
2. **92% là tỷ lệ MRR của Random so với Full phải không?** Không. Đó là tỷ lệ phần tăng so với CLIP còn giữ lại.
3. **Có thể kết luận cạnh thật chắc chắn chỉ đóng góp 8% không?** Không. Đây là tỷ lệ mô tả một lượt chạy, không phải phân rã nhân quả và chưa có độ bất định đa seed.

## Slide 8 — AdaMF và đồ thị tri thức đa phương thức

**Tiêu đề trên slide:** MMKG Adds Relations Beyond the Image.

### Lời thuyết trình

Scene graph chỉ liên kết đối tượng trong từng ảnh. Với truy vấn như tìm người có quan hệ với một nhân vật, nhóm cần đồ thị tri thức liên kết các thực thể ngoài nội dung trực tiếp của ảnh.

Ở cấp độ 2, AdaMF dung hợp ba nguồn: embedding cấu trúc, văn bản và hình ảnh. Khi thiếu một phương thức, nhóm che phương thức đó trước softmax để nó không nhận trọng số như thông tin có thật. Biểu diễn dung hợp được dùng trong bộ chấm điểm RotatE và gán xuống các ảnh của thực thể ứng viên.

DB15K có 12.842 thực thể và 25.925 ảnh; FB15K-237-IMG có 14.541 thực thể và 139.814 ảnh. Nhóm thích nghi AdaMF-MAT cho truy hồi ảnh, không chạy nguyên bài toán gốc; thành phần MAT được tắt. DB15K dùng validation sạch tách từ official train để chọn mô hình.

### Giải thích kiến thức

- **AdaMF:** dung hợp đa phương thức thích nghi bằng trọng số học được. Trọng số lớn không tự chứng minh một phương thức là nguyên nhân quyết định; cần can thiệp để kiểm tra.
- **RotatE:** biểu diễn quan hệ như phép quay trong không gian phức; bộ ba phù hợp khi biểu diễn đầu sau phép quay gần biểu diễn đuôi.
- Truy vấn $(h,r,?)$ tìm đuôi; $(?,r,t)$ tìm đầu. Neo và quan hệ được cung cấp trong bản ghi truy vấn, không giả định mô hình tự trích chính xác từ văn bản tùy ý.
- **Modality mask:** loại phương thức thiếu khỏi phép phân bổ attention. Vector 0 thay thế dữ liệu thiếu không phải ảnh thật.
- MAT bị tắt khác với lấy mẫu âm tự đối kháng của loss KGC; đây là hai cơ chế khác nhau.

### Câu hỏi dự kiến và trả lời ngắn

1. **Ảnh và thực thể có phải một đối tượng không?** Không. Một thực thể có thể có nhiều ảnh; đầu ra retrieval xếp hạng ảnh, còn điểm KG được tính ở cấp thực thể.
2. **Nhóm có tái lập đầy đủ AdaMF-MAT của bài báo không?** Không nên tuyên bố như vậy. Nhóm thích nghi AdaMF, tắt MAT và đánh giá theo giao thức truy hồi ảnh của đồ án.
3. **Vì sao không dùng official validation DB15K?** Kiểm định phát hiện trùng lặp và chồng lấn với train/test; nhóm tạo validation sạch từ train, giữ official test.

## Slide 9 — Kết quả trực tiếp và gián tiếp

**Tiêu đề trên slide:** Indirect Queries Drive the Gains on Both MMKGs.

### Lời thuyết trình

Ở nhóm truy vấn gián tiếp, DB15K tăng MRR từ khoảng 0,0475 lên 0,3143, tương đương 6,6 lần. FB15K tăng từ khoảng 0,0122 lên 0,1186, tương đương 9,7 lần. Những tỷ lệ này mô tả mức của mô hình so với baseline, không phải xác suất truy vấn đúng tăng tương ứng.

Với DB15K, truy vấn trực tiếp không thay đổi vì pipeline chủ động giữ thứ hạng CLIP. Với FB15K, nhóm trực tiếp tăng từ khoảng 0,3475 lên 0,7770; nhánh này dùng mục tiêu truy hồi và học các thành phần biểu diễn, nên không thể xem đó là cùng một cơ chế với DB15K.

Đối chứng KG-GAT trên cùng pipeline DB15K cho MRR gián tiếp gần AdaMF và khoảng tin cậy chênh lệch chứa 0. Nhóm chưa tìm thấy khác biệt rõ trong phép đánh giá này, không kết luận hai mô hình tương đương tuyệt đối.

### Giải thích kiến thức

- Chỉ số tổng hợp phụ thuộc tỷ trọng direct/indirect. Báo riêng hai nhóm giúp nhận ra nguồn gốc mức tăng.
- “Tăng lên 6,6 lần” là $MRR_{mới}/MRR_{gốc}$; cách nói “tăng thêm 6,6 lần” dễ gây nhầm với phần chênh lệch.
- KG-GAT là mô hình được thích nghi cho cùng pipeline KG, không phải lấy nguyên checkpoint scene graph Visual Genome rồi áp trực tiếp.
- Khoảng tin cậy chứa 0: dữ liệu chưa loại trừ khả năng chênh lệch bằng 0; không phải kiểm định tương đương.
- Không so 0,3143 của DB15K với 0,1186 của FB15K để kết luận DB15K có mô hình tốt hơn, vì bài toán và tập ứng viên khác nhau.

### Câu hỏi dự kiến và trả lời ngắn

1. **Direct DB15K không tăng có phải mô hình thất bại không?** Không. Nhánh direct giữ nguyên CLIP theo thiết kế, nên không được dùng để đánh giá lợi ích của reranking đồ thị ở nhánh đó.
2. **FB15K tăng direct có chứng minh cạnh tốt hơn không?** Chưa. Pipeline đồng thời học lớp chiếu, biểu diễn dung hợp và mục tiêu truy hồi.
3. **AdaMF và GAT có bằng nhau không?** Kết quả gần nhau và chưa phát hiện khác biệt rõ trong đánh giá này; không chứng minh bằng nhau hoặc tương đương trên mọi dữ liệu.

## Slide 10 — Vai trò cạnh thật trên hai benchmark KG

**Tiêu đề trên slide:** Edge Corruption Hurts More on DB15K than FB15K.

### Lời thuyết trình

Biểu đồ dùng tỷ lệ lợi ích MRR gián tiếp so với CLIP còn giữ lại sau can thiệp. Trên DB15K, No Edges bỏ huấn luyện và dùng CLIP. Random Graph được validation chọn alpha bằng 1 nên điểm đồ thị không tham gia thứ hạng cuối. Shuffled Edges chỉ giữ khoảng 37% lợi ích.

Trên FB15K, random và shuffled vẫn giữ khoảng 93% lợi ích. Can thiệp ở đây thay neo của truy vấn train gián tiếp, trong khi giám sát trực tiếp và các tham số dùng chung vẫn tồn tại. Vì vậy, mô hình còn nhiều cách học liên kết thực thể–ảnh mà không cần đúng các liên kết đã bị phá.

Kết luận của nhóm là đối chứng hỗ trợ đóng góp của quan hệ đúng trên DB15K, còn FB15K chưa cô lập lợi ích của cấu trúc. Cần nhớ mỗi biến thể mới chạy một lần và Shuffled có dùng held-out triples để loại cạnh trùng.

### Giải thích kiến thức

- DB15K Shuffled hoán đổi đuôi trong cùng quan hệ, bảo toàn bậc theo quan hệ; FB15K Shuffled hoán đổi neo trong nhóm quan hệ và hướng truy vấn train.
- 37% và 93% là phần **lợi ích vượt CLIP**, không phải tỷ lệ dự đoán đúng hoặc tỷ lệ cạnh còn nguyên.
- Random chọn alpha bằng 1 không chứng minh đồ thị ngẫu nhiên vô hại khi buộc dùng; quy trình lựa chọn đã bỏ điểm đó.
- **Test-aware exclusion:** đọc danh sách held-out để tránh sinh cạnh trùng. Nó khác sử dụng test để tối ưu metric, nhưng không hoàn toàn test-blind.
- Khác biệt giữa hai bộ còn gồm loss, chiều embedding và ngân sách huấn luyện; không quy toàn bộ cho một yếu tố.

### Câu hỏi dự kiến và trả lời ngắn

1. **93% có nghĩa FB15K không cần đồ thị trong mọi trường hợp không?** Không. Nó cho thấy phần lớn lợi ích còn tồn tại dưới những can thiệp cụ thể trong một lượt chạy.
2. **No Edges ở đây giống No-edge của Visual Genome không?** Không. Hai bộ KG bỏ huấn luyện và về CLIP; Visual Genome vẫn giữ thành phần học được.
3. **Shuffled có hoàn toàn độc lập với test không?** Không. Nó dùng held-out triples để loại cạnh trùng; nhóm công bố giới hạn này và không gọi là sinh đồ thị mù hoàn toàn với test.

## Slide 11 — Sinh giải thích bằng can thiệp

**Tiêu đề trên slide:** Explanations Come From the Ranking Function.

### Lời thuyết trình

Nhóm giải thích ảnh đang đứng hạng 1 của hệ thống, kể cả khi ảnh đó sai. Với từng đơn vị bằng chứng, nhóm can thiệp và đo thay đổi điểm. GAT có thể bỏ quan hệ rồi chạy lại mô hình; với AdaMF, nhóm học lại embedding cấu trúc cục bộ để xấp xỉ ảnh hưởng của bộ ba, giữ các thành phần khác cố định.

Các phần tử có đóng góp dương cao được chọn thành tập giải thích. Số lượng tối đa K được chọn trên validation theo mức độ bao phủ tác động can thiệp, không chọn dựa vào test.

Nhóm đo Fid+ khi bỏ giải thích, Fid− khi chỉ giữ giải thích, PN/PS về vị trí top 1, độ thưa và tính hợp lệ cấu trúc. Đây là attribution bằng can thiệp sau huấn luyện, không phải khẳng định mô hình tự sinh suy luận đúng về thế giới.

### Giải thích kiến thức

- Gọi $p_0$ là xác suất softmax của ảnh đang giải thích trong tập ứng viên; $p_{-S}$ là xác suất sau khi bỏ tập $S$, $p_S$ là sau khi chỉ giữ $S$.
- $Fid^+=(p_0-p_{-S})/p_0$: càng lớn thì bỏ bằng chứng làm ảnh giảm xác suất càng mạnh. Giá trị có thể âm nếu ảnh tăng xác suất.
- $Fid^-=(p_0-p_S)/p_0$: mức giảm nhỏ thể hiện giữ bằng chứng vẫn duy trì được xác suất; giá trị âm nghĩa xác suất tăng, không phải tỷ lệ đúng âm.
- PN đo tỷ lệ ảnh trả về mất top 1 khi bỏ $S$; PS đo tỷ lệ còn top 1 khi chỉ giữ $S$. Chúng theo dự đoán của mô hình, không phải theo đáp án đúng.
- Completeness dùng để chọn K là tỷ lệ mức giảm khi bỏ top-K so với bỏ tập bằng chứng nền được quy định trong thuật toán; ngưỡng 0,8 không có nghĩa giải thích đúng 80% hay đủ ở mọi khía cạnh.
- Phương pháp vẫn mang tính post-hoc. Mạnh hơn đường đi đẹp về hình thức ở chỗ có kiểm tra tác động lên hàm chấm điểm; với AdaMF, tác động này còn là xấp xỉ.

### Câu hỏi dự kiến và trả lời ngắn

1. **Tại sao không lấy attention lớn nhất làm giải thích?** Attention không tự đo ảnh hưởng của cạnh lên điểm của từng truy vấn; nhóm dùng can thiệp và xem attention là đối chứng khi phù hợp.
2. **Bỏ bộ ba có phải chỉ xóa dòng khỏi dữ liệu rồi chấm checkpoint cũ không?** Không. Thông tin đã nằm trong embedding; AdaMF phải dùng học lại cục bộ hoặc phép huấn luyện lại phù hợp để đo tác động.
3. **Ngưỡng completeness 0,8 có chứng minh tính đủ không?** Không. Nó chọn độ dài theo tiêu chí tác động khi bỏ; tính đủ cần đo thêm bằng chỉ giữ tập giải thích, Fid− và PS.

## Slide 12 — Kết quả XAI và giới hạn

**Tiêu đề trên slide:** Fidelity Separates Explanations From Random Removal.

### Lời thuyết trình

Biểu đồ so sánh bỏ tập giải thích với bỏ cùng số phần tử ngẫu nhiên trên năm cặp mô hình–dữ liệu: GAT Visual Genome, AdaMF DB15K, AdaMF FB15K và hai mô hình trên ViHeritage. Trung bình tác động khi bỏ giải thích lớn hơn đối chứng ở cả năm cặp; không có nghĩa mọi truy vấn riêng lẻ đều như vậy.

Validity cấu trúc đạt 100% trong các mẫu đã kiểm tra, còn sparsity nằm khoảng 0,65 đến 0,80 tùy bộ giải thích. Những con số này không đồng nghĩa mọi giải thích đều đúng ngữ nghĩa.

Trên ViHeritage, kiểm chứng AdaMF bằng huấn luyện lại cho Fid+ 0,327, cao hơn 0,180 của bỏ ngẫu nhiên, với p bằng 0,0002. Tuy nhiên, tương quan Spearman giữa tác động xấp xỉ và huấn luyện lại chỉ 0,32. Kết quả hỗ trợ hướng tác động ở mức tổng hợp, nhưng yêu cầu thận trọng khi diễn giải từng truy vấn.

### Giải thích kiến thức

- Đối chứng cần bỏ cùng số phần tử để tránh giải thích có tác động lớn chỉ vì xóa nhiều dữ liệu hơn.
- **Structural validity:** bằng chứng là quan hệ hợp lệ trong scene graph hoặc bộ ba train không chồng lấn held-out theo kiểm tra của từng pipeline.
- **Sparsity:** tỷ lệ phần tử không giữ lại so với tập nền được định nghĩa. Trên KG, đó không phải toàn bộ hàng nghìn cạnh của toàn KG.
- **Spearman 0,32:** thứ tự độ ảnh hưởng giữa hai phép đo chỉ tương quan hạn chế; không phải độ chính xác XAI 32%.
- $p=0,0002$ không phải xác suất kết luận sai 0,02%; nó mô tả mức cực đoan của thống kê dưới giả thuyết không theo phép kiểm định.
- Xác suất softmax dùng làm đại lượng chẩn đoán, không mặc định là xác suất dự đoán đúng đã hiệu chuẩn.

### Câu hỏi dự kiến và trả lời ngắn

1. **Validity 100% có nghĩa giải thích nào cũng đúng không?** Không. Chỉ tính hợp lệ cấu trúc được xác nhận trong mẫu; tính đúng và liên quan ngữ nghĩa cần đánh giá khác.
2. **Fidelity cao có thể xuất hiện với dự đoán sai không?** Có. Bằng chứng có thể ảnh hưởng mạnh đến một quyết định sai của mô hình.
3. **Spearman 0,32 có làm toàn bộ phép xấp xỉ vô giá trị không?** Không, nhưng giới hạn khả năng dùng xấp xỉ để khẳng định tác động chính xác ở từng truy vấn; cần báo cùng đối chứng huấn luyện lại.

## Slide 13 — Dữ liệu và kết quả ViHeritage-MMKG

**Tiêu đề trên slide:** ViHeritage-MMKG: A Vietnamese Heritage Graph.

### Lời thuyết trình

ViHeritage tập trung vào di sản và danh thắng Việt Nam, có 150 thực thể, trong đó 88 có ảnh, 15 quan hệ, 427 bộ ba và 438 ảnh. Nguồn chính là Wikidata, Wikipedia tiếng Việt và Wikimedia Commons; giấy phép được ghi theo từng nguồn và từng ảnh.

Trên 54 truy vấn gián tiếp test tiếng Việt, MRR tăng từ 0,112 của CLIP lên 0,219 với AdaMF và 0,258 với GAT. Hai phương pháp đều cải thiện so với CLIP, nhưng chưa có bằng chứng rõ rằng GAT tốt hơn AdaMF trên test.

Khi che văn bản thực thể, AdaMF đạt 0,230; nhóm chưa quan sát thấy suy giảm trong thí nghiệm này. Xáo cạnh làm kết quả thấp hơn CLIP, hỗ trợ vai trò liên kết đúng trong các lượt chạy đã xét. Đây là áp dụng và huấn luyện phương pháp trên miền đích, không phải chuyển nguyên checkpoint mà không thích nghi.

### Giải thích kiến thức

- Bản chia bộ ba: 318 train, 53 validation, 56 test; bản chia ảnh: 264/86/88. Bộ ba và ảnh có đơn vị khác nhau, không cộng lẫn.
- Có 88 truy vấn trực tiếp và 54 gián tiếp test. MRR gián tiếp ở đây không phải MRR trên toàn bộ 142 truy vấn.
- “Audit 0,935” là chỉ số kiểm tra liên kết ảnh trong mẫu 100 ảnh theo data card, có hỗ trợ AI và giới hạn đánh giá; không phải 93,5% ảnh toàn bộ đã được người chấm độc lập xác nhận. Không suy thành đúng chính xác 93,5 trên 100 ảnh.
- Khoảng 35% bộ ba được suy từ thuộc tính như vị trí và thời điểm xây dựng, không phải tất cả đều là phát biểu quan hệ có sẵn trực tiếp.
- Che văn bản không chứng minh hoàn toàn không có leakage; nó kiểm tra sự phụ thuộc vào phương thức văn bản trong cấu hình này.
- Text dùng encoder đa ngôn ngữ; dữ liệu nhỏ và thiên về địa danh nổi tiếng nên phạm vi tổng quát hóa còn hạn chế.

### Câu hỏi dự kiến và trả lời ngắn

1. **Vì sao 150 thực thể nhưng chỉ 88 có ảnh?** Các thực thể còn lại vẫn cần để tạo quan hệ, như tỉnh, khái niệm và nhân vật; không phải mọi nút đều là ứng viên ảnh.
2. **Che văn bản mà kết quả cao hơn có chứng minh văn bản vô ích không?** Không. Chênh lệch chưa cho bằng chứng rõ về cải thiện; chỉ chưa quan sát thấy suy giảm ở thí nghiệm đang xét.
3. **Đây có phải zero-shot transfer không?** Không. Phương pháp được cấu hình và huấn luyện trên ViHeritage; không tuyên bố chuyển nguyên checkpoint sang các ID thực thể mới.

## Slide 14 — Tổng hợp điều bằng chứng hỗ trợ

**Tiêu đề trên slide:** What the Evidence Supports — and What It Does Not.

### Lời thuyết trình

Nhóm tách kết luận thành phần được bằng chứng hỗ trợ và phần chưa thể khẳng định. Các pipeline có học đồ thị cải thiện so với CLIP trong những giao thức đã thử, đặc biệt ở nhóm truy vấn gián tiếp. Đối chứng DB15K và ViHeritage hỗ trợ đóng góp của quan hệ đúng. Bằng chứng XAI được chọn cũng nhạy với can thiệp hơn đối chứng ngẫu nhiên ở mức tổng hợp.

Ngược lại, trên Visual Genome phần lớn mức tăng vẫn còn khi bỏ hoặc xáo cạnh; trên FB15K, can thiệp vẫn giữ khoảng 93% lợi ích. Nhóm không quy toàn bộ cải thiện cho suy luận quan hệ. Chênh lệch AdaMF và GAT trên DB15K chưa rõ về thống kê, không phải đã chứng minh hai phương pháp bằng nhau.

Thông điệp chung là phải kiểm tra đúng nơi đồ thị có đóng góp, đồng thời phân biệt ảnh hưởng tới mô hình với tính đúng theo kiến thức thực tế.

### Giải thích kiến thức

- **Bằng chứng tích cực:** quan sát phù hợp với giả thuyết trong phạm vi dữ liệu và giao thức; không tự trở thành định luật tổng quát.
- **Không tìm thấy khác biệt:** chưa đủ bằng chứng bác bỏ chênh lệch bằng 0; kiểm định tương đương là một thiết kế khác.
- Không dùng MRR giữa các dataset để so trực tiếp độ mạnh của phương pháp, vì khác tập ứng viên, truy vấn và relevance.
- Ablation giúp kiểm tra cách giải thích kết quả, không chỉ giúp tạo thêm bảng số liệu.
- Khi nói “đồ thị giúp”, phải làm rõ đang nói toàn bộ pipeline có thành phần đồ thị hay vai trò riêng của liên kết đúng.

### Câu hỏi dự kiến và trả lời ngắn

1. **Kết luận quan trọng nhất của nhóm là gì?** Đồ thị có ích trong các giao thức đã khảo sát, nhưng lợi ích của pipeline và lợi ích riêng của cạnh đúng cần được kiểm tra tách biệt.
2. **Có thể nói mô hình học được suy luận quan hệ ở cả bốn dataset không?** Chưa. Đối chứng cho thấy mức phụ thuộc vào liên kết đúng rất khác nhau giữa các pipeline.
3. **Kết quả nào còn cần bổ sung nhất?** Ablation nhiều seed, đánh giá ngữ nghĩa độc lập và kiểm chứng sát mô hình gốc hơn cho phép xấp xỉ XAI.

## Slide 15 — Kết luận và chuyển sang demo

**Tiêu đề trên slide:** Conclusions & Live Demo.

### Lời thuyết trình

Nhóm xin kết lại bằng ba ý. Thứ nhất, CLIP kết hợp GAT cải thiện truy hồi Visual Genome khoảng 17,4% MRR tương đối, nhưng phần lớn mức tăng vẫn tồn tại khi bỏ cạnh. Thứ hai, AdaMF cải thiện đáng kể truy vấn gián tiếp; đối chứng hỗ trợ vai trò quan hệ đúng trên DB15K, còn FB15K chưa tách được đóng góp riêng đó. Thứ ba, nhóm có giải thích bằng can thiệp và bộ dữ liệu di sản tiếng Việt, đồng thời công bố giới hạn của phép xấp xỉ và đánh giá ngữ nghĩa.

Sau đây, nhóm chuyển sang demo. Nhóm sẽ lần lượt cho xem CLIP và reranking, một truy vấn gián tiếp, tác động khi bỏ bằng chứng so với bỏ ngẫu nhiên, rồi một truy vấn tiếng Việt trên ViHeritage. Khi trình bày, nhóm phân biệt rõ thao tác chạy mới với kết quả đã lưu. Nếu môi trường trực tiếp gặp lỗi, nhóm dùng bản ghi dự phòng đã chuẩn bị.

### Giải thích kiến thức

- Mục tiêu demo là nối kết quả định lượng với một trường hợp quan sát được; một case đẹp không thay thế bảng đánh giá trên tập test.
- **Luồng demo:** nhập truy vấn → hiển thị kết quả CLIP → reranking → xem ảnh hạng 1 và bằng chứng → can thiệp và xem thay đổi.
- Trước demo cần chuẩn bị đúng checkpoint, dữ liệu, môi trường và query có thể tái lập. Không chỉnh alpha hoặc K sau khi nhìn kết quả.
- Với AdaMF, cần công bố bước nào là xấp xỉ học lại và bước nào dùng cache. Không nói “vừa huấn luyện lại toàn bộ” nếu chỉ đọc artifact.
- Video, notebook và khả năng chạy trực tiếp là hạng mục phải kiểm tra thực tế; slide mô tả kế hoạch không tự chứng minh đã hoàn thành.

### Câu hỏi dự kiến và trả lời ngắn

1. **Demo một truy vấn có đủ chứng minh hiệu năng không?** Không. Demo minh họa hành vi; bằng chứng hiệu năng là các bảng kết quả với giao thức và mẫu đánh giá đã công bố.
2. **Nếu chạy live lỗi thì làm thế nào?** Mở bản ghi hoặc kết quả đã lưu, nói rõ đó là dữ liệu dự phòng và trình bày cùng cấu hình đã khóa.
3. **Nếu chỉ cải tiến thêm một việc thì nhóm chọn gì?** Ưu tiên tăng độ chắc chắn của kết luận: ablation nhiều seed và kiểm chứng XAI sát mô hình gốc hơn.

## Slide 16 / B1 — Chỉ số và giao thức đánh giá

**Tiêu đề trên slide:** Backup: Metrics and Evaluation Protocol.

### Lời thuyết trình

Slide này làm rõ cách nhóm đo kết quả. Recall@K tính tỷ lệ truy vấn có ít nhất một ảnh liên quan trong top K. MRR lấy trung bình nghịch đảo hạng của ảnh liên quan tốt nhất, nên phản ánh chất lượng đưa kết quả đúng lên đầu. Với các bộ KG, nhóm báo riêng direct và indirect để không che khuất ảnh hưởng của đồ thị.

Về kiểm soát dữ liệu, Visual Genome gom ảnh trùng và gần trùng trước khi chia. DB15K bỏ official validation có vấn đề và chia lại official train theo tỷ lệ 90/10. FB15K xử lý cụm ảnh trùng trước chia; đặc trưng thị giác dùng trong huấn luyện được gộp từ ảnh train.

Nhóm cũng công bố ngoại lệ: Shuffled dùng held-out triples để loại cạnh trùng nên không hoàn toàn test-blind. Với ViHeritage, đối chứng che văn bản kiểm tra ảnh hưởng của tên neo trong mô tả đáp án.

### Giải thích kiến thức

- $Recall@K=\frac{1}{|Q|}\sum_q\mathbf{1}[rank_q\le K]$; $MRR=\frac{1}{|Q|}\sum_q1/rank_q$.
- Khi một truy vấn có nhiều ảnh đúng, hạng dùng cho MRR là hạng ảnh đúng tốt nhất theo hợp đồng đánh giá của bộ dữ liệu.
- **Optimistic ranking:** $rank_q=1+\#\{i:S_i>S_{đúng\ tốt\ nhất}\}$. Điểm bằng nhau không được tính là cao hơn; cần giữ quy tắc nhất quán giữa các lượt so sánh.
- **SHA-256:** phát hiện trùng nội dung tệp theo hash; **pHash:** hỗ trợ tìm ảnh gần giống về thị giác. Không có nghĩa mọi ảnh khác hash đều khác hẳn nội dung.
- Seed chia dữ liệu khác seed huấn luyện. Giữ một bản chia chung rồi thay seed mô hình giúp so sánh điều kiện học tương ứng.
- Độ lệch chuẩn mẫu qua ba seed dùng mẫu số $n-1$; không gọi đó là sai số chuẩn hoặc khoảng tin cậy nếu chưa thực hiện chuyển đổi tương ứng.

### Câu hỏi dự kiến và trả lời ngắn

1. **Nếu một thực thể có nhiều ảnh đúng thì chấm thế nào?** Các ảnh hợp lệ của thực thể đích trong đúng split là positive; MRR dùng ảnh liên quan tốt nhất và Recall@K cần ít nhất một ảnh đúng trong top K.
2. **Vì sao gom ảnh trùng trước chia?** Để cùng nội dung ảnh không xuất hiện ở cả train và test, làm đánh giá dễ hơn một cách không mong muốn.
3. **Đã loại trừ hoàn toàn mọi dạng leakage chưa?** Không nên khẳng định tuyệt đối. Nhóm kiểm soát nhiều dạng cụ thể và công bố giới hạn như test-aware exclusion, ảnh gần trùng và tín hiệu văn bản.

## Slide 17 / B2 — Bộ 20 truy vấn minh họa cấp độ 2

**Tiêu đề trên slide:** Backup: The 20-Query Evaluation Set (Level 2).

### Lời thuyết trình

Bên cạnh đánh giá toàn bộ test, nhóm xây một bộ 20 truy vấn cho mỗi dataset để kiểm tra yêu cầu của đồ án và minh họa giải thích. Bộ này có mười truy vấn trực tiếp và mười gián tiếp; phần gián tiếp gồm năm head và năm tail. Mỗi truy vấn có một thực thể đích khác nhau, được chọn bằng quy tắc cố định trước khi xem kết quả.

Trên DB15K, MRR của mười truy vấn gián tiếp tăng từ 0,003 lên 0,296; nhóm trực tiếp giữ nguyên theo thiết kế. Trên FB15K, direct tăng mạnh nhưng indirect trong mẫu nhỏ chỉ tăng từ 0,012 lên 0,025.

Sự khác biệt với bảng toàn test cho thấy một bộ nhỏ có thể không đại diện đầy đủ cho phân phối truy vấn. Vì vậy, nhóm dùng 20 truy vấn để minh họa và kiểm tra, còn bảng toàn test là bằng chứng hiệu năng chính.

### Giải thích kiến thức

- **Head query:** che đầu trong $(?,r,t)$; **tail query:** che đuôi trong $(h,r,?)$.
- Một truy vấn cho mỗi thực thể đích giảm lặp thực thể trong tập nhỏ, nhưng không bảo đảm tập nhỏ đại diện cho mọi quan hệ hoặc mức độ khó.
- Với số lượng direct/indirect bằng nhau trong mẫu, MRR gộp là trung bình có trọng số 1/2 của hai nhóm, trước làm tròn.
- Trên toàn test, tỷ trọng hai nhóm khác 1/2 nên không lấy trung bình đơn giản hai MRR nhóm để suy ra MRR chung.
- Không dùng bộ 20 truy vấn để chỉnh tham số rồi tiếp tục gọi kết quả trên nó là đánh giá độc lập.
- Mẫu nhỏ vẫn có thể được phân tích thống kê khi phương pháp phù hợp; giới hạn ở đây là độ đại diện và độ bất định, không phải một lệnh cấm kiểm định tuyệt đối.

### Câu hỏi dự kiến và trả lời ngắn

1. **Vì sao kết quả indirect FB15K ở bộ 20 thấp hơn nhiều so với toàn test?** Mẫu chỉ có mười truy vấn gián tiếp với độ khó riêng; không nên xem nó là ước lượng chính xác của toàn phân phối.
2. **Nhóm có chọn các truy vấn sau khi thấy mô hình thắng không?** Theo giao thức, truy vấn được chọn bằng quy tắc và seed cố định trước khi xem kết quả, không chọn theo mức cải thiện.
3. **Bộ 20 truy vấn có thay thế đánh giá toàn test không?** Không. Nó phục vụ minh họa và kiểm tra yêu cầu; đánh giá hiệu năng chính dùng toàn test.

## Slide 18 / B3 — Case Visual Genome: “peace sign”

**Tiêu đề trên slide:** Backup: Case Study 1 — Visual Genome.

### Lời thuyết trình

Truy vấn của case này là “Guy holding up peace sign”. CLIP xếp ảnh đúng ở hạng 15, còn GAT rerank đưa lên hạng 1. Khi bỏ tập giải thích, ảnh tụt xuống hạng 4; No-edge và Random-edges trong case này lần lượt cho hạng 4 và 2.

Quan sát đóng góp cho thấy cạnh contains tới các nút “peace sign” chiếm ưu thế. Đóng góp của từng cạnh này khoảng 5,5 nhân 10 mũ trừ 2, trong khi quan hệ holding up khoảng 0,2 nhân 10 mũ trừ 2, nhỏ hơn khoảng 27 lần.

Như vậy, case thành công chủ yếu được hỗ trợ bởi sự có mặt của đối tượng khớp truy vấn, không chứng minh mô hình đã hiểu đầy đủ quan hệ hành động. Cách giải thích này phù hợp với kết quả ablation, nhưng một case không thay thế kết luận trên toàn tập.

### Giải thích kiến thức

- Trong hình, nút ảnh màu cam; nút đối tượng màu xanh. Nhãn cạnh kèm giá trị đóng góp theo quy ước thang đo đã ghi trên hình.
- Sự xuất hiện nhiều nút cùng nhãn có thể đến từ chú giải nhiều đối tượng hoặc vùng; không mặc định coi năm nút là năm sự kiện suy luận độc lập.
- Đóng góp từng cạnh được tính bằng can thiệp riêng. Không cộng cơ học các giá trị để suy ra tác động bỏ cả tập, vì GAT phi tuyến và các cạnh có tương tác.
- Kết quả No-edge hoặc Random-edges theo một case có thể khác thứ tự trung bình trên toàn test.
- “Tụt từ 1 xuống 4” vẫn là ảnh đúng trong top 5; tác động giải thích không bắt buộc làm đáp án biến mất hoàn toàn.

### Câu hỏi dự kiến và trả lời ngắn

1. **Vì sao nhiều cạnh contains có ảnh hưởng lớn hơn quan hệ holding up?** Trong case này, mô hình ưu tiên tín hiệu đối tượng khớp truy vấn; điều đó không chứng minh mọi truy vấn đều như vậy.
2. **Bỏ giải thích mà ảnh vẫn hạng 4 thì giải thích có tốt không?** Nó có ảnh hưởng đến thứ hạng; đánh giá chất lượng còn cần so với bỏ ngẫu nhiên, tính đủ và độ gọn.
3. **Có thể cộng năm mức giảm cạnh contains không?** Không trực tiếp. Can thiệp từng phần tử và can thiệp cả tập có thể khác do tương tác phi tuyến.

## Slide 19 / B4 — Case DB15K: Hitchens và Dawkins

**Tiêu đề trên slide:** Backup: Case Study 2 — DB15K Indirect Query.

### Lời thuyết trình

Trong case này, truy vấn tìm ảnh của thực thể có quan hệ influenced by hướng tới Richard Dawkins. Hệ thống đưa ảnh Christopher Hitchens từ hạng CLIP 55 lên hạng 1 sau reranking AdaMF.

Nhóm đo riêng ảnh hưởng của phương thức theo quy ước delta bằng điểm gốc trừ điểm sau che. Che cấu trúc làm điểm giảm khoảng 74,8; che văn bản giảm khoảng 0,20; che ảnh cho delta khoảng âm 0,15, nghĩa là điểm tăng nhẹ. Vì vậy, kết quả này nhạy với phương thức cấu trúc hơn hai phương thức còn lại.

Bộ giải thích chọn tối đa hai bộ ba. Một bằng chứng nổi bật được lưu trong artifact là Hitchens–influenced–Dawkins, có đóng góp xấp xỉ 0,198. Cần phân biệt tác động che toàn phương thức với tác động của một bộ ba, đồng thời kiểm tra chiều và ý nghĩa quan hệ trước khi coi đường đi là phát biểu tri thức đúng.

### Giải thích kiến thức

- **Neo:** Dawkins; **đích:** Hitchens. Mô hình chấm ứng viên điền vào vị trí đầu bị thiếu của quan hệ trong truy vấn.
- Delta dương theo công thức $\Delta=S_{orig}-S_{masked}$ nghĩa là che làm giảm điểm. Delta âm nghĩa là che làm tăng điểm; không tự suy ra phương thức đó luôn có hại.
- 74,8 là mức thay đổi điểm trong phép che phương thức của case này, không phải 74,8% Fidelity hay thay đổi MRR.
- 0,198 là đóng góp của một bộ ba theo phép xấp xỉ học lại cục bộ; không cùng phép can thiệp với che toàn bộ cấu trúc.
- Nhãn influenced trong bằng chứng và influenced by trong truy vấn không được mặc định hoán đổi tự do. Cần kiểm tra ID, chiều quan hệ và ngữ nghĩa dữ liệu.
- Mức điểm thay đổi lớn không tự là bằng chứng đã giải thích đúng mọi bước bên trong embedding gốc.

### Câu hỏi dự kiến và trả lời ngắn

1. **Delta ảnh âm có nghĩa ảnh không cần thiết trong mô hình không?** Không. Nó chỉ cho thấy che ảnh làm điểm tăng nhẹ ở ứng viên và truy vấn này.
2. **74,8 và 0,198 có so trực tiếp để tính tỷ lệ đóng góp không?** Không. Chúng đến từ hai phép can thiệp khác nhau: che phương thức và xấp xỉ ảnh hưởng một bộ ba.
3. **Đường đi có cạnh là đã đúng ngữ nghĩa chưa?** Chưa. Cần kiểm tra nhãn và chiều quan hệ; tính hợp lệ trong train không bảo đảm tri thức hoặc diễn giải ngôn ngữ đúng.

## Slide 20 / B5 — Case ViHeritage: di tích ở Huế

**Tiêu đề trên slide:** Backup: Case Study 3 — ViHeritage Vietnamese MMKG.

### Lời thuyết trình

Với câu hỏi “Thành phố Huế có những di tích nào?”, CLIP xếp ảnh Chùa Thiên Mụ ở hạng 3; cả GAT và AdaMF đưa ảnh này lên hạng 1.

Một đường đi được minh họa là Chùa Thiên Mụ thuộc Quần thể di tích Cố đô Huế, và quần thể này nằm ở Huế. Trong hình, Huế là neo màu xanh ngọc, Chùa Thiên Mụ là thực thể ảnh màu cam, còn các cạnh đậm là bằng chứng được chọn. Với GAT, tập giải thích có tối đa tám phần tử; Fid+ của case khoảng 0,49 so với 0,01 khi bỏ ngẫu nhiên.

Case này minh họa cách kết quả xếp hạng và tri thức địa phương có thể được trình bày cùng nhau. Nó không tự chứng minh hiệu quả trên toàn bộ di sản Việt Nam; bằng chứng tổng hợp vẫn là bảng đánh giá trên tập test.

### Giải thích kiến thức

- Truy vấn tự nhiên được gắn với một khung neo–quan hệ trong dữ liệu; không tuyên bố hệ thống đã giải quyết tổng quát việc phân tích mọi câu tiếng Việt tự do.
- Đường đi hai bước hỗ trợ diễn giải câu trả lời, nhưng đánh giá tính hợp lý cần xét cả chiều cạnh và ngữ nghĩa của quan hệ trung gian.
- GAT có thể dùng nhiều cạnh trong vùng tiếp nhận; một đường đi minh họa không nhất thiết là toàn bộ bằng chứng làm thay đổi điểm.
- $K^\star=8$ là giới hạn đã chọn trên validation, không phải số cạnh bắt buộc mọi truy vấn đều có và không phải độ dài đường đi tám bước.
- Dùng mô hình trên tiếng Việt ở đây có encoder văn bản đa ngôn ngữ và huấn luyện trên dữ liệu đích; không quy thành khả năng tiếng Việt sẵn có của mọi encoder CLIP.

### Câu hỏi dự kiến và trả lời ngắn

1. **K bằng 8 có nghĩa đường suy luận dài tám bước không?** Không. Đó là số phần tử tối đa trong tập bằng chứng; đường đi minh họa có thể chỉ gồm hai bước.
2. **Vì sao chọn ảnh Chùa Thiên Mụ, còn Huế không phải ảnh đáp án?** Huế là thực thể neo của câu hỏi; Chùa Thiên Mụ là một thực thể đáp án phù hợp cần được tìm bằng quan hệ.
3. **Case này chứng minh chuyển giao hoàn toàn chưa?** Nó minh họa áp dụng phương pháp trên dữ liệu đích; kết luận hiệu quả cần dựa vào toàn bộ test và giao thức huấn luyện đã công bố.

## Slide 21 / B6 — Các trường hợp thất bại

**Tiêu đề trên slide:** Backup: Failure-Case Analysis.

### Lời thuyết trình

Phân tích thất bại giúp nhóm hiểu mô hình đang dựa vào tín hiệu gì. Trên Visual Genome, một số ảnh sai có đối tượng khớp truy vấn nhưng sai thuộc tính hoặc quan hệ. Có ba trên năm case thất bại được khảo sát mà No-edge vẫn xếp ảnh đúng đầu tiên; sau message passing kết quả lại sai.

Trên DB15K, nhóm gặp quan hệ gần nghĩa gây nhầm, bộ ba nhiễu và lỗi trả về chính thực thể neo. Với FB15K, một số thực thể hub có hàng nghìn bộ ba khiến đóng góp từng bộ ba rất nhỏ; văn bản đôi khi quan trọng hơn cấu trúc trong phép che phương thức.

ViHeritage cũng có lỗi trả về neo ở truy vấn “ở gần X”, bằng chứng đúng nhưng không liên quan và tri thức nguồn sai. Các tỷ lệ trên slide chỉ mô tả những case đã chọn, không phải tỷ lệ lỗi trên toàn test.

### Giải thích kiến thức

- **Graph hurts:** CLIP đã xếp đúng tốt nhưng reranking làm tệ hơn. **Not rescued:** mô hình không đưa được đáp án lên vị trí mong muốn; không đồng nghĩa đồ thị nhất thiết làm hại.
- **Hub:** thực thể có nhiều cạnh. Nhiều bộ ba không mặc định mang nhiều thông tin hữu ích cho một truy vấn cụ thể.
- **Đúng nhưng không liên quan:** bộ ba có thể đúng ngoài đời mà không hỗ trợ quan hệ được hỏi; đây là vấn đề relevance của giải thích.
- Lỗi dữ liệu có thể được mô hình học và đưa vào bằng chứng. Giải thích giúp phát hiện lỗi, không bảo đảm loại lỗi tự động.
- “6/10 trả về neo” trên ViHeritage là quan sát trong các case thất bại được phân tích của hai mô hình, không phải 60% toàn bộ truy vấn test.

### Câu hỏi dự kiến và trả lời ngắn

1. **Tại sao có thêm đồ thị mà kết quả kém hơn?** Tín hiệu quan hệ có thể nhiễu, không liên quan hoặc lấn át đặc trưng hữu ích; mô hình cũng có thể học liên kết chưa phù hợp.
2. **Có nên loại thực thể neo khỏi mọi truy vấn không?** Không áp dụng máy móc. Chỉ loại khi ngữ nghĩa truy vấn và quan hệ yêu cầu một thực thể khác; phải xác định quy tắc trước đánh giá.
3. **Có thể dùng các case này để tính tỷ lệ lỗi chung không?** Không. Đây là mẫu được chọn để chẩn đoán; muốn ước lượng tỷ lệ chung cần mẫu và thiết kế phù hợp.

## Slide 22 / B7 — Cấu hình huấn luyện

**Tiêu đề trên slide:** Backup: Training Configuration.

### Lời thuyết trình

Bảng này cho thấy các pipeline dùng chung nền CLIP nhưng không giống nhau hoàn toàn về huấn luyện. Visual Genome học GAT bằng cross-entropy với hard negatives, tối đa 100 epoch. DB15K học mục tiêu bộ ba với 128 mẫu âm, dim quan hệ 250 và tối đa 500 epoch. FB15K dùng dim 64 và mục tiêu truy hồi với 50 hard negatives.

Trên ViHeritage, hai cấu hình được chọn qua validation: AdaMF dùng dim 50 và learning rate 5 nhân 10 mũ trừ 4, còn GAT dùng dim 100 và learning rate 2 nhân 10 mũ trừ 3. Cả hai có batch 64, 32 mẫu âm và giới hạn 1.500 epoch, kèm dừng sớm.

Do nhiều yếu tố khác nhau, nhóm chỉ so sánh các đối chứng trong đúng giao thức tương ứng; không quy khác biệt kết quả giữa dataset cho riêng loại loss hoặc mô hình.

### Giải thích kiến thức

- **Learning rate:** độ lớn bước cập nhật; batch size là số mẫu trong một lượt tối ưu, không phải số ảnh ứng viên khi đánh giá.
- **Negative sampling:** lấy mẫu bộ ba hoặc ảnh không phù hợp để học phân biệt. “128 negatives” DB15K và “50 hard negatives” VG không cùng đơn vị bài toán.
- **Self-adversarial weighting:** tăng trọng số cho mẫu âm đang khó theo điểm mô hình; không đồng nghĩa bật MAT.
- Trong RotatE của cấu hình này, dim là chiều embedding quan hệ; biểu diễn thực thể gồm phần thực và ảo có chiều $2\times dim$. DB15K tương ứng 500 chiều; FB15K 128 chiều.
- Max epochs là trần huấn luyện, không phải mọi checkpoint đều ở epoch cuối. Patience gắn với số lần đánh giá validation không cải thiện theo lịch đã định, không mặc định là từng ấy epoch.
- Số seed mô hình đầy đủ, seed chia dữ liệu và một lượt chạy ablation cần được ghi riêng khi tái lập.

### Câu hỏi dự kiến và trả lời ngắn

1. **Vì sao DB15K có dim 250 nhưng thực thể 500 chiều?** Vì RotatE dùng hai phần thực–ảo cho thực thể; dim quan hệ và chiều vector lưu của thực thể không giống nhau.
2. **MAT bằng 0 có nghĩa không còn lấy mẫu âm đối kháng không?** Không. MAT và cách gán trọng số self-adversarial cho mẫu âm là hai cơ chế khác nhau.
3. **Có thể kết luận loss KGC tốt hơn cross-entropy từ hai dataset không?** Không. Dữ liệu, chiều embedding, can thiệp và ngân sách học đều khác; cần thí nghiệm kiểm soát riêng.

## Slide 23 / B8 — Phân công và sản phẩm bàn giao

**Tiêu đề trên slide:** Backup: Team Contributions.

### Lời thuyết trình

Nhóm phân công công việc theo các thành phần nhưng phối hợp ở bước tích hợp và kiểm chứng. Khánh phụ trách kiến trúc tổng thể, audit và biên tập; Hạ phụ trách tiền xử lý, kiểm soát rò rỉ và đặc trưng CLIP; Uyên phụ trách scene graph, GAT và phân tích cấp độ 1; Hiển triển khai AdaMF và các đối chứng cấp độ 2; Nghiệp phụ trách giải thích, đánh giá XAI và dữ liệu MMKG tự xây.

Các sản phẩm cần bàn giao gồm mã nguồn, môi trường và hướng dẫn tái lập, dữ liệu và bản chia được mô tả rõ, báo cáo, slide, bộ truy vấn, phân tích case, data card và demo. Nhóm công bố việc sử dụng mã nguồn mở và hỗ trợ AI; công cụ hỗ trợ không thay thế trách nhiệm kiểm tra và bảo vệ nội dung của thành viên.

Khi xác nhận mức hoàn thành, nhóm đối chiếu từng sản phẩm thực tế thay vì chỉ dựa vào checklist trên slide.

### Giải thích kiến thức

- Bảng phân công mô tả trách nhiệm chính, không khẳng định một người thực hiện độc lập mọi công đoạn liên quan.
- **Tái lập:** cần biết dữ liệu nào, bản chia nào, cấu hình nào, checkpoint nào và lệnh nào tạo ra kết quả; chỉ có notebook chưa đủ.
- **Data card:** mô tả nguồn, cách thu thập, quy mô, chia tập, giấy phép và giới hạn dữ liệu; khác với log huấn luyện.
- Sử dụng công cụ AI để hỗ trợ viết truy vấn hoặc chấm thử ngữ nghĩa cần được công bố. Một lượt chấm có AI hỗ trợ không được trình bày như đánh giá độc lập nhiều người.
- Checklist có nhắc giới hạn trang báo cáo và video theo đề bài; phải đối chiếu yêu cầu chính thức và file bàn giao thực tế trước khi nói đã đáp ứng tất cả. Không lấy chữ trên slide làm bằng chứng hoàn thành.

### Câu hỏi dự kiến và trả lời ngắn

1. **Nhóm có dùng AI không?** Có, theo phạm vi đã công bố như hỗ trợ mã, ngôn ngữ truy vấn và biên tập; thành viên phải kiểm tra và chịu trách nhiệm về kết quả.
2. **Làm sao người khác chạy lại kết quả?** Dùng hướng dẫn repo, dữ liệu đã xử lý đúng phiên bản, môi trường phù hợp và checkpoint/cấu hình tương ứng; phân biệt chạy lại đánh giá với huấn luyện lại.
3. **Data card giúp gì ngoài việc ghi nguồn?** Nó làm rõ cách dữ liệu được tạo, giới hạn, giấy phép và giao thức đánh giá, giúp người dùng không suy kết luận vượt phạm vi dữ liệu.

## Ghi nhớ khi luyện nói và trả lời

- Luôn phân biệt **pipeline có học đồ thị cải thiện** với **cạnh đúng là nguyên nhân của toàn bộ mức tăng**.
- 77%, 92%, 37% và khoảng 93% là **tỷ lệ phần lợi ích so với CLIP còn giữ lại**, không phải tỷ lệ ảnh đúng.
- DB15K và ViHeritage giữ direct theo CLIP; FB15K có học và cải thiện direct. Không nói cả ba có nhánh trực tiếp giống nhau.
- XAI giải thích ảnh mô hình trả về, kể cả ảnh sai. Structural validity không bằng semantic validity; fidelity không bằng ground-truth correctness.
- AdaMF học lại cục bộ là xấp xỉ. Spearman 0,32 trên kiểm chứng ViHeritage không phải 32% giải thích đúng; Pearson điểm gốc–mimic và Spearman tác động can thiệp đo hai điều khác nhau.
- Kích thước tập giải thích không phải độ dài đường đi. K được chọn trên validation, không chỉnh lại cho đẹp ở case test.
- Các giá trị hiển thị được làm tròn; khi tính tỷ lệ tăng hoặc đối chiếu sai khác nhỏ, dùng artifact có độ chính xác gốc.
- Không suy tỷ lệ lỗi toàn test từ 5 case thành công và 5 case thất bại. Không dùng CI chứa 0 để tuyên bố hai mô hình tương đương.
- Khi cần chỉ vào hình: nói **truy vấn → hạng trước/sau → phần tử nổi bật → tác động can thiệp → giới hạn**. Không đọc tất cả nhãn cạnh.
- Chỉ mở slide dự phòng phù hợp câu hỏi; không trình bày cả tám slide như một phần bắt buộc sau phần kết luận.

