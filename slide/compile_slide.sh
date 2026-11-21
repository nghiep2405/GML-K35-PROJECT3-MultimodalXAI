#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
BUILD_SUBDIR="slide_compiled/${BUILD_TIMESTAMP}"
BUILD_DIR="${SCRIPT_DIR}/${BUILD_SUBDIR}"
MAIN_TEX="main_slide.tex"

build_suffix=1
while [[ -e "${BUILD_DIR}" ]]; do
  build_suffix=$((build_suffix + 1))
  BUILD_SUBDIR="slide_compiled/${BUILD_TIMESTAMP}_${build_suffix}"
  BUILD_DIR="${SCRIPT_DIR}/${BUILD_SUBDIR}"
done

for required_command in pdflatex; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Lỗi: không tìm thấy ${required_command}. Hãy cài đặt trước khi biên dịch." >&2
    exit 1
  fi
done

mkdir -p "${BUILD_DIR}"
cd "${SCRIPT_DIR}"

run_pdflatex() {
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -jobname="${BUILD_SUBDIR}/main_slide" \
    "${MAIN_TEX}"
}

echo "=== Đang biên dịch Slide Beamer (Lần 1) ==="
run_pdflatex

echo "=== Đang biên dịch Slide Beamer (Lần 2) ==="
run_pdflatex

# Sao chép bản build mới nhất ra file main_slide.pdf và output/final_presentation.pdf
cp "${BUILD_DIR}/main_slide.pdf" "${SCRIPT_DIR}/main_slide.pdf" 2>/dev/null || true
mkdir -p "${SCRIPT_DIR}/output"
cp "${BUILD_DIR}/main_slide.pdf" "${SCRIPT_DIR}/output/final_presentation.pdf" 2>/dev/null || true

echo "Biên dịch slide hoàn tất: ${BUILD_DIR}/main_slide.pdf"
echo "Bản xuất chính thức: ${SCRIPT_DIR}/output/final_presentation.pdf"
