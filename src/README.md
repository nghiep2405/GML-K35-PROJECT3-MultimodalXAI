# Mã nguồn Multimodal XAI

Cài package từ thư mục gốc bằng `python -m pip install -e "./src[test]"`. Lệnh `multimodal-xai local <pipeline> <action>` tự tìm repo và đọc trực tiếp `our_run_result/`. Trên Colab, `multimodal-xai drive <pipeline> <action>` dùng `/content/drive/MyDrive/MultimodalXAI/` làm gốc với cùng bundle. Kết quả mới ghi vào `runs/`.

Xem [hướng dẫn cấu trúc và cách chạy](../structure_instruction__repo.md) để chọn dependency và lệnh từng pipeline. Visual Genome tải riêng theo README trong `our_run_result/data/VisualGeome/`.
