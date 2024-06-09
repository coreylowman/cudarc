#!/bin/bash

set -exu

bindgen \
  --allowlist-type "^cuda.*" \
  --allowlist-type "^surfaceReference" \
  --allowlist-type "^textureReference" \
  --allowlist-var "^cuda.*" \
  --allowlist-function "^cuda.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  wrapper.h -- -/usr/local/cuda/include \
  > tmp.rs

CUDART_VERSION=$(cat tmp.rs | grep "CUDART_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs sys_${CUDART_VERSION}.rs
