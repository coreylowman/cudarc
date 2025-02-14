if [ -z "${CUDA_ROOT}" ]; then
    CUDA_DIR=$CUDA_ROOT
elif [ -d "/opt/cuda" ]; then
    CUDA_DIR="/opt/cuda"
else
    CUDA_DIR="/usr/local/cuda"
fi

export CUDA_INCLUDES="${CUDA_DIR}/include"

declare -a modules=("cublas" "cublaslt" "cudnn" "curand" "driver" "runtime" "nccl" "nvrtc")
for path in "${modules[@]}"; do
    cd src/${path}/sys
    bash bindgen.sh
    cd ../../../
done
