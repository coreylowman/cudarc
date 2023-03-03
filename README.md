# cudarc: minimal and safe api over the cuda toolkit

Safe abstractions over:
1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
4. [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)

**Pre-alpha state**, expect breaking changes and not all cuda functions
contain a safe wrapper. **Contributions welcome for any that aren't included!**

```rust
let dev = cudarc::driver::CudaDevice::new(0)?;

// allocate buffers
let inp = dev.htod_copy(vec![1.0f32; 100])?;
let mut out = dev.alloc_zeros::<f32>(100)?;

// compile our kernel
let ptx = cudarc::nvrtc::compile_ptx("
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}")?;

// load it into the device
dev.load_ptx(ptx, "my_module", &["sin_kernel"])?;

// launch it!
let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
let cfg = LaunchConfig::for_num_elems(100);
unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100usize)) }?;

// and finally copy back to host
let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
assert_eq!(out_host, [1.0; 100].map(f32::sin));
```

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
