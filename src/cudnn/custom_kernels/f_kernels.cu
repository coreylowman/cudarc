// every function must end with `f32` and only accept `float`s; then the same function with `f64` and `double`s will be generated

extern "C" __global__ void division_f32(float *out, const float *a, const float *b, size_t numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = a[i] / b[i];
    }
}