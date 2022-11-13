// every function must end with `f32` and only accept `float`s; then the same function with `f64` and `double`s will be generated

extern "C" __global__ void division_with_scale_f32(float *out, const float *a, const float *b, const float *scale, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = scale[0] * a[i] / b[i];
    }
}

extern "C" __global__ void division_f32(float *out, const float *a, const float *b, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = a[i] / b[i];
    }
}