# Project 03 — Multimodal XAI

Đề bài của nhóm nằm trong [Project03_Multimodal_XAI.pdf](Project03_Multimodal_XAI.pdf).

| Thư mục | Nội dung |
| --- | --- |
| [doc/](doc/) | Ghi chú, lưu ý và tài liệu tham khảo trong quá trình làm đồ án. |
| [report/](report/) | Mã nguồn LaTeX, hình ảnh và tài liệu tham khảo cho báo cáo cuối kỳ. |
| [src/](src/) | Mã nguồn triển khai đồ án. |

## Biên dịch báo cáo

Cần cài đặt bộ LaTeX có `pdfLaTeX` và `BibTeX`. Từ thư mục gốc của repo, chạy script:

```bash
./report/compile.sh
```

Hoặc chạy từ thư mục `report`:

```bash
cd report
./compile.sh
```

Script thực hiện lệnh biên dịch tương đương:

```bash
BUILD_TIME="$(date '+%Y-%m-%d_%H-%M-%S')"
mkdir -p "report_compiled/${BUILD_TIME}"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="report_compiled/${BUILD_TIME}/main" main.tex
bibtex "report_compiled/${BUILD_TIME}/main"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="report_compiled/${BUILD_TIME}/main" main.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="report_compiled/${BUILD_TIME}/main" main.tex
```

Mỗi lần biên dịch sẽ tạo một thư mục riêng theo thời gian, ví dụ `report/report_compiled/2026-11-14_16-30-00/`. PDF nằm tại `main.pdf` trong thư mục đó. Các bản build cũ được giữ nguyên và toàn bộ `report/report_compiled/` được Git bỏ qua.
