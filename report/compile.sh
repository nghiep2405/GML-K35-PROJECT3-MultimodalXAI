#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_TEX_BIN="${SCRIPT_DIR}/../.local/TinyTeX/bin/universal-darwin"
BUILD_SUBDIR="report_compiled/build"
BUILD_DIR="${SCRIPT_DIR}/${BUILD_SUBDIR}"
MAIN_TEX="main.tex"
PUBLISHED_PDF="${SCRIPT_DIR}/Project03_Report.pdf"

if [[ -x "${LOCAL_TEX_BIN}/pdflatex" ]]; then
  export PATH="${LOCAL_TEX_BIN}:${PATH}"
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Lỗi: không tìm thấy pdflatex. Hãy cài TinyTeX/MacTeX trước khi biên dịch." >&2
  exit 1
fi

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

if [[ -f "${BUILD_DIR}/main.bcf" ]]; then
  if ! command -v biber >/dev/null 2>&1; then
    echo "Lỗi: tài liệu dùng biblatex nhưng không tìm thấy biber." >&2
    exit 1
  fi
  biber "${BUILD_SUBDIR}/main"
else
  if ! command -v bibtex >/dev/null 2>&1; then
    echo "Lỗi: không tìm thấy bibtex." >&2
    exit 1
  fi
  bibtex "${BUILD_SUBDIR}/main"
fi

run_pdflatex
run_pdflatex

cp "${BUILD_DIR}/main.pdf" "${PUBLISHED_PDF}"

echo "Biên dịch hoàn tất: ${BUILD_DIR}/main.pdf"
echo "Bản PDF để chia sẻ: ${PUBLISHED_PDF}"
