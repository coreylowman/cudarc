extern "C" __global__ void division_f32(float *out, const float *a, const float *b, size_t numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = a[i] / b[i];
    }
}
extern "C" __global__ void division_f64(double *out, const double *a, const double *b, size_t numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = a[i] / b[i];
    }
}