# Kết quả kiểm thử

- `baseline/source-manifest.json` ghi SHA-256 của source gốc, gồm cả notebook lịch sử.
- `baseline/manifest.json` ghi SHA-256 của các bảng đối chiếu từ lần chạy cũ.
- `runs/<run-id>/` lưu log, JUnit, JSON tóm tắt, preflight và kết quả thực thi. Có thể tái tạo thư mục này bằng các lệnh trong [hướng dẫn repo](../structure_instruction__repo.md).

`PASS` nghĩa là bước đã chạy và đạt. `BLOCKED` nghĩa là thiếu dependency hoặc artifact. Khi báo cáo replay, ghi checkpoint/hash đầu vào và so theo cùng query ID, seed và protocol.
