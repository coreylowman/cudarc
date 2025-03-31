#!/bin/bash
set -exu

bindgen \
    --allowlist-var="^CUDA_VERSION.*" \
    --allowlist-type="^nvrtc.*" \
    --allowlist-function="^nvrtc.*" \
    --default-enum-style=rust \
    --no-doc-comments \
    --with-derive-default \
    --with-derive-eq \
    --with-derive-hash \
    --with-derive-ord \
    --use-core \
    --dynamic-loading Lib \
    wrapper.h -- -I$CUDA_INCLUDES \
    >tmp.rs

CUDA_VERSION=$(cat tmp.rs | grep "CUDA_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs loading/sys_${CUDA_VERSION}.rs

bindgen \
    --allowlist-var="^CUDA_VERSION.*" \
    --allowlist-type="^nvrtc.*" \
    --allowlist-function="^nvrtc.*" \
    --default-enum-style=rust \
    --no-doc-comments \
    --with-derive-default \
    --with-derive-eq \
    --with-derive-hash \
    --with-derive-ord \
    --use-core \
    wrapper.h -- -I$CUDA_INCLUDES \
    >tmp.rs

CUDA_VERSION=$(cat tmp.rs | grep "CUDA_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs linked/sys_${CUDA_VERSION}.rs
