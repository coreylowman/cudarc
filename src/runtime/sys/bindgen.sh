#!/bin/bash

set -exu

bindgen \
  --allowlist-type "^[Cc][Uu][Dd][Aa].*" \
  --allowlist-var "^[Cc][Uu][Dd][Aa].*" \
  --allowlist-function "^[Cc][Uu][Dd][Aa].*" \
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

CUDART_VERSION=$(cat tmp.rs | grep "CUDART_VERSION" | awk '{ print $6 }' | sed 's/.$//')
mv tmp.rs sys_${CUDART_VERSION}.rs

if [[ "$OSTYPE" == "msys" ]]; then
    # windows
    nvcc -shared -Xcompiler -fPIC -o libtestkernel.dll testkernel.cu
    mv libtestkernel.dll ../../../target/debug/
else
    # linux
    nvcc -shared -Xcompiler -fPIC -o libtestkernel.so testkernel.cu
    mv libtestkernel.so ../../../target/debug/
fi
