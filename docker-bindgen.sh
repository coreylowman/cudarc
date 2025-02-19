apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y curl clang
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
cargo install bindgen-cli@0.71.1

export CUDA_INCLUDES="/usr/local/cuda/include"

declare -a modules=("cublas" "cublaslt" "cudnn" "curand" "driver" "runtime" "nccl" "nvrtc")
for path in "${modules[@]}"; do
    cd src/${path}/sys
    bash bindgen.sh
    cd ../../../
done
