#!/bin/bash
set -exu

bindgen \
  --allowlist-var="^CUDA_VERSION.*" \
  --allowlist-type="^cudnn.*" \
  --allowlist-var="^cudnn.*" \
  --allowlist-function="^cudnn.*" \
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
