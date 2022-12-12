#!/bin/bash
set -exu

bindgen \
  --allowlist-type "^(cudnn|CUDNN).*" \
  --allowlist-var "^(cudnn|CUDNN).*" \
  --allowlist-function "^(cudnn|CUDNN).*" \
  --no-doc-comments \
  --default-enum-style=rust \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  --use-core \
  --output sys.rs \
  wrapper.h -- --include-directory "/mnt/c/Program Files/NVIDIA/CUDNN/v8.6/include" --include-directory "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include"