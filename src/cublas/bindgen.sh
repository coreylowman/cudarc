#!/bin/bash
set -exu

bindgen \
  --whitelist-type="^cublas.*" \
  --whitelist-var="^cublas.*" \
  --whitelist-function="^cublas.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  --use-core \
  wrapper.h -- --include-directory="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include" \
  > sys.rs