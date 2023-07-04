#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^nccl.*" \
  --allowlist-var="^nccl.*" \
  --allowlist-function="^nccl.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/usr/local/cuda/include -I/usr/lib/gcc/x86_64-linux-gnu/12/include/ \
  > sys.rs
