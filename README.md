# Project 03 — Multimodal XAI

Đề bài của nhóm nằm trong [Project03_Multimodal_XAI.pdf](Project03_Multimodal_XAI.pdf).

| Thư mục | Nội dung |
| --- | --- |
| [doc/](doc/) | Ghi chú, lưu ý và tài liệu tham khảo trong quá trình làm đồ án. |
| [report/](report/) | Mã nguồn LaTeX, hình ảnh và tài liệu tham khảo cho báo cáo cuối kỳ. |
| [src/](src/) | Mã nguồn triển khai đồ án. |

## Biên dịch báo cáo

Chạy từ thư mục gốc của repo với LaTeX có XeLaTeX và latexmk.

```powershell
cd report
latexmk -xelatex main.tex
```

Kết quả nằm tại `report/main.pdf`.
