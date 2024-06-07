declare -a modules=("cublas" "cublaslt" "cudart" "cudnn" "curand" "driver" "nccl" "nvrtc")
for path in "${modules[@]}"
do
    cd src/${path}/sys
    bash bindgen.sh
    cd ../../../
done
