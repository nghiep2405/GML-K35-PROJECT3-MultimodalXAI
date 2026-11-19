# Visual Genome

Bộ dữ liệu Visual Genome đã preprocess có dung lượng khoảng 12 GB và được tải riêng từ Google Drive.

[Tải dữ liệu Visual Genome đã preprocess](https://drive.google.com/drive/folders/1qg9C6RcVA467jkjOj6-0tSuF1vLmUXJj?usp=drive_link)

Giữ nguyên cấu trúc bên trong gói tải xuống. Code hiện tại dùng các thư mục đầu vào sau, tính từ thư mục gốc repo hoặc thư mục MultimodalXAI trên Drive.

- `dataset/visual_genome/processed/`
- `dataset/visual_genome/graph/split_v2/`
- `dataset/visual_genome/clip_baseline/split_v2/`
- `dataset/visual_genome/model/gat_v1/cache/`
- `experiment_v2/data/`

Thư mục `VisualGeome/` này ghi vị trí tải dữ liệu. Đường dẫn `visual_genome` trong code phải giữ đúng tên khi bố trí đầu vào. Xem `structure_instruction__repo.md` ở gốc repo để chạy các component.

Liên kết được nhóm cung cấp. Chưa kiểm tra truy cập hoặc nội dung gói dữ liệu trong lần rà soát này.
