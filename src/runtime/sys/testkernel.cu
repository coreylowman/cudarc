#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>

extern "C" __global__ void halfs(__half h) {
    assert(__habs(h - __float2half(1.234)) <= __float2half(1e-4));
}

extern "C" __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}

extern "C" __global__ void int_8bit(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    assert(s_min == -128);
    assert(s_max == 127);
    assert(u_min == 0);
    assert(u_max == 255);
}

extern "C" __global__ void int_16bit(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    assert(s_min == -32768);
    assert(s_max == 32767);
    assert(u_min == 0);
    assert(u_max == 65535);
}

extern "C" __global__ void int_32bit(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    assert(s_min == -2147483648);
    assert(s_max == 2147483647);
    assert(u_min == 0);
    assert(u_max == 4294967295);
}

extern "C" __global__ void int_64bit(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    assert(s_min == -9223372036854775808);
    assert(s_max == 9223372036854775807);
    assert(u_min == 0);
    assert(u_max == 18446744073709551615);
}

extern "C" __global__ void floating(float f, double d) {
    assert(fabs(f - 1.2345678) <= 1e-7);
    assert(fabs(d - -10.123456789876543) <= 1e-16);
}

extern "C" __global__ void slow_worker(const float *data, const size_t len, float *out) {
    float tmp = 0.0;
    for(size_t i = 0; i < 1000000; i++) {
        tmp += data[i % len];
    }
    *out = tmp;
}

// wrappers
extern "C" void halfs_wrapper(__half h) {
    halfs<<<1, 1>>>(h);
}

extern "C" void sin_kernel_wrapper(float *out, const float *inp, size_t numel) {
    const size_t block_size = 256;
    const size_t grid_size = (numel + block_size - 1) / block_size;
    sin_kernel<<<grid_size, block_size>>>(out, inp, numel);
}

extern "C" void int_8bit_wrapper(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    int_8bit<<<1, 1>>>(s_min, s_max, u_min, u_max);
}

extern "C" void int_16bit_wrapper(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    int_16bit<<<1, 1>>>(s_min, s_max, u_min, u_max);
}

extern "C" void int_32bit_wrapper(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    int_32bit<<<1, 1>>>(s_min, s_max, u_min, u_max);
}

extern "C" void int_64bit_wrapper(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    int_64bit<<<1, 1>>>(s_min, s_max, u_min, u_max);
}

extern "C" void floating_wrapper(float f, double d) {
    floating<<<1, 1>>>(f, d);
}

extern "C" void slow_worker_wrapper(const float *data, const size_t len, float *out) {
    slow_worker<<<1, 1>>>(data, len, out);
}

extern "C" void slow_worker_stream_wrapper(const float *data, const size_t len, float *out, cudaStream_t stream) {
    slow_worker<<<1, 1, 0, stream>>>(data, len, out);
}
