// every function must end with `f32` and only accept `float`s; then the same function with `f64` and `double`s will be generated

// TODO only use one scale parameter
extern "C" __global__ void division_with_scale_f32(float *out, const float *a, const float *a_scale, const float *b, const float *b_scale, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = (a_scale[0] / b_scale[0]) * a[i] / b[i];
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

extern "C" __global__ void sin_with_scale_f32(float *out, const float *a, const float *scale, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = sin(a[i]) * scale[0];
    }
}

extern "C" __global__ void sin_f32(float *out, const float *a, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = sin(a[i]);
    }
}

extern "C" __global__ void cos_with_scale_f32(float *out, const float *a, const float *scale, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = cos(a[i]) * scale[0];
    }
}

extern "C" __global__ void cos_f32(float *out, const float *a, size_t numel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)
    {
        out[i] = cos(a[i]);
    }
}