# Biên dịch report

Chạy các lệnh sau tại thư mục gốc repo.

## Windows / PowerShell

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\report\compile.ps1
```

Script đọc vị trí TinyTeX từ `.local/tex-environment.json` nếu có, hoặc tìm
tại `.local/TinyTeX/bin/windows`, rồi chạy pdfLaTeX,
BibTeX và thêm hai lượt pdfLaTeX để cập nhật tài liệu tham khảo, số bảng và
tham chiếu. Đường dẫn có dấu được chuyển sang đường dẫn ngắn Windows trong
quá trình chạy; không cần đổi tên hay di chuyển repo.

## Git Bash, Linux và macOS

```bash
bash report/compile.sh
```

Script dò TinyTeX trong `.local/TinyTeX/bin/*`, hoặc dùng `pdflatex` và
`bibtex` đã có trong `PATH`. Trên Windows có thể dùng Git Bash đi kèm Git
for Windows; nếu Bash bị môi trường chặn, dùng lệnh PowerShell ở trên.

## Công cụ và đầu ra

- TinyTeX dành cho đúng hệ điều hành: <https://yihui.org/tinytex/>.
- Máy hiện tại cài TinyTeX tại
  `C:\Users\ADMIN\AppData\Local\Temp\gml-latex-tools\TinyTeX` để tránh lỗi
  đường dẫn Unicode của TeX. `.local/tex-environment.json` lưu vị trí này.
  Nếu dọn thư mục Temp làm mất công cụ, cần cài lại TinyTeX và cập nhật
  `texRoot` trong tệp cấu hình, hoặc thêm bản TeX đã cài vào `PATH`.
- Thư mục `.local/` được Git bỏ qua; môi trường của máy này không tự xuất
  hiện khi clone repo trên máy khác. Máy mới cần cài TinyTeX/TeX Live với
  các gói `ieeetran`, `vntex` cùng những gói được khai báo trong `main.tex`.
- PDF cập nhật: `report/Project03_Report.pdf`.
- PDF trung gian và log: `report/report_compiled/build/main.pdf` và `main.log`.
- Nguồn chính: `report/main.tex`; nội dung chương trong `report/Chapter*/`;
  tài liệu tham khảo trong `report/References/references.bib`.

Sửa nội dung `.tex` rồi chạy lại cùng lệnh. PDF công bố chỉ được thay thế sau
khi các lượt biên dịch và xử lý tài liệu tham khảo thành công.
