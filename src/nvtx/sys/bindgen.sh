#!/bin/bash
set -exu

# Follow https://github.com/rust-lang/rust-bindgen/discussions/2405
bindgen \
  --allowlist-var="^CUDA_VERSION.*" \
  --allowlist-type="^nvtx.*" \
  --allowlist-function="^nvtx.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  --wrap-static-fns \
  --experimental \
  wrapper.h -- -I/usr/local/cuda/include \
  > tmp.rs

mv /tmp/bindgen/extern.c .

CUDA_VERSION=$(cat tmp.rs | grep "CUDA_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs sys_${CUDA_VERSION}.rs
