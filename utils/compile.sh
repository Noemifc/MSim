#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# CUDA compile script for liblamino.so
# Usage:
#   CUDA_SRC_DIR=../cuda \         # path to CUDA sources
#   LAM_KERNEL_SRC=lkernel.cu \   # CUDA source filename
#   LAM_OUT_LIB=libmsim.so \    # output library name
#   NVCC=nvcc \                   # CUDA compiler
#   ./compile.sh
# ----------------------------------------------------------------------------

# Default directories and filenames (override via env vars)
: "${CUDA_SRC_DIR:=../msim/cuda}"
: "${LAM_KERNEL_SRC:=lkernel.cu}"
: "${LAM_OUT_LIB:=libmsim.so}"
: "${NVCC:=nvcc}"

SRC_PATH="${CUDA_SRC_DIR}/${LAM_KERNEL_SRC}"
OUT_LIB="${CUDA_SRC_DIR}/${LAM_OUT_LIB}"

echo "[LSIM] Compiling CUDA kernel from: ${SRC_PATH}"
echo "[LSIM] Output library will be: ${OUT_LIB}"

# Create output directory if needed
mkdir -p "${CUDA_SRC_DIR}"

# Compile with position-independent code and as a shared library
"/usr/local/cuda-11.0/bin/nvcc" \
    --compiler-options "-fPIC" \
    -shared \
    -o "${OUT_LIB}" "${SRC_PATH}" \
    -lcudart

# Verify
if [[ -f "${OUT_LIB}" ]]; then
    echo "[LSIM] Successfully built ${OUT_LIB}"
else
    echo "[LSIM] ERROR: Failed to build ${OUT_LIB}" >&2
    exit 1
fi
