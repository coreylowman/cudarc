#!/bin/bash
set -exu

bindgen \
    --allowlist-var="^CUDA_VERSION" \
    --allowlist-type="^cublas.*" \
    --allowlist-function="^cublas.*" \
    --default-enum-style=rust \
    --no-doc-comments \
    --with-derive-default \
    --with-derive-eq \
    --with-derive-hash \
    --with-derive-ord \
    --use-core \
    --dynamic-loading Lib \
    --no-layout-tests \
    wrapper.h -- -I$CUDA_INCLUDES \
    >tmp.rs

CUDA_VERSION=$(cat tmp.rs | grep "CUDA_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs sys_${CUDA_VERSION}.rs
