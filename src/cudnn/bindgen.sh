#!/bin/bash
set -exu

bindgen \
  --whitelist-type="^cudnn.*" \
  --whitelist-var="^cudnn.*" \
  --whitelist-function="^cudnn.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  --use-core \
  wrapper.h -- --include-directory "/mnt/c/Program Files/NVIDIA/CUDNN/include" --include-directory "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include" \
  > sys.rs