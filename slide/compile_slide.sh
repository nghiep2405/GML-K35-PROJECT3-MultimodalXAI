#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
BUILD_SUBDIR="slide_compiled/${BUILD_TIMESTAMP}"
BUILD_DIR="${SCRIPT_DIR}/${BUILD_SUBDIR}"
TARGET="${1:-all}"

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

compile_file() {
  local tex_file="$1"
  local job_name
  job_name="$(basename "${tex_file}" .tex)"

  echo "=== Đang biên dịch ${tex_file} (Lần 1) ==="
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -jobname="${BUILD_SUBDIR}/${job_name}" \
    "${tex_file}"

  echo "=== Đang biên dịch ${tex_file} (Lần 2) ==="
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -jobname="${BUILD_SUBDIR}/${job_name}" \
    "${tex_file}"

  # Sao chép bản build mới nhất ra thư mục gốc slide và output/
  mkdir -p "${SCRIPT_DIR}/output"
  cp "${BUILD_DIR}/${job_name}.pdf" "${SCRIPT_DIR}/${job_name}.pdf" 2>/dev/null || true
  if [[ "${job_name}" == "main_slide" ]]; then
    cp "${BUILD_DIR}/${job_name}.pdf" "${SCRIPT_DIR}/output/final_presentation.pdf" 2>/dev/null || true
  elif [[ "${job_name}" == "main_slide_vi" ]]; then
    cp "${BUILD_DIR}/${job_name}.pdf" "${SCRIPT_DIR}/output/final_presentation_vi.pdf" 2>/dev/null || true
  fi

  echo "Biên dịch hoàn tất: ${BUILD_DIR}/${job_name}.pdf -> ${SCRIPT_DIR}/${job_name}.pdf"
}

if [[ "${TARGET}" == "all" ]]; then
  compile_file "main_slide.tex"
  compile_file "main_slide_vi.tex"
else
  compile_file "${TARGET}"
fi
