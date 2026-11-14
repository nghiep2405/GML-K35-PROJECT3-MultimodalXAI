#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
BUILD_SUBDIR="report_compiled/${BUILD_TIMESTAMP}"
BUILD_DIR="${SCRIPT_DIR}/${BUILD_SUBDIR}"
MAIN_TEX="main.tex"

build_suffix=1
while [[ -e "${BUILD_DIR}" ]]; do
  build_suffix=$((build_suffix + 1))
  BUILD_SUBDIR="report_compiled/${BUILD_TIMESTAMP}_${build_suffix}"
  BUILD_DIR="${SCRIPT_DIR}/${BUILD_SUBDIR}"
done

for required_command in pdflatex bibtex; do
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
    -jobname="${BUILD_SUBDIR}/main" \
    "${MAIN_TEX}"
}

run_pdflatex
bibtex "${BUILD_SUBDIR}/main"
run_pdflatex
run_pdflatex

echo "Biên dịch hoàn tất: ${BUILD_DIR}/main.pdf"
