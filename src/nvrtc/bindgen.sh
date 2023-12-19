#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^nvrtc.*" \
  --allowlist-function="^nvrtc.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  --dynamic-loading Nvrtc \
  wrapper.h -- -I/usr/local/cuda/include \
  > sys.rs
