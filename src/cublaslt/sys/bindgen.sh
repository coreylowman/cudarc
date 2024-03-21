#!/bin/bash
# Requires rust-bindgen 0.68.1 or superior
set -exu
BINDGEN_EXTRA_CLANG_ARGS="-D__CUDA_BF16_TYPES_EXIST__" \
bindgen \
  --allowlist-var="^CUDA_VERSION.*" \
  --allowlist-type="^cublasLt.*" \
  --allowlist-var="^cublasLt.*" \
  --allowlist-function="^cublasLt.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  --dynamic-loading Lib \
  wrapper.h -- -I/usr/local/cuda/include \
  > tmp.rs

CUDA_VERSION=$(cat tmp.rs | grep "CUDA_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs sys_${CUDA_VERSION}.rs
