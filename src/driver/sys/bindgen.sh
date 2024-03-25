#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^CU.*" \
  --allowlist-type="^cuuint(32|64)_t" \
  --allowlist-type="^cudaError_enum" \
  --allowlist-type="^cu.*Complex$" \
  --allowlist-type="^cuda.*" \
  --allowlist-type="^libraryPropertyType.*" \
  --allowlist-var="^CU.*" \
  --allowlist-function="^cu.*" \
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
