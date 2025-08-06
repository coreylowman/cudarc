# cudarc: minimal and safe api over the cuda toolkit

[![](https://dcbadge.vercel.app/api/server/AtUhGqBDP5)](https://discord.gg/AtUhGqBDP5)
[![crates.io](https://img.shields.io/crates/v/cudarc?style=for-the-badge)](https://crates.io/crates/cudarc)
[![docs.rs](https://img.shields.io/docsrs/cudarc?label=docs.rs%20latest&style=for-the-badge)](https://docs.rs/cudarc)

Checkout cudarc on [crates.io](https://crates.io/crates/cudarc) and [docs.rs](https://docs.rs/cudarc/latest/cudarc/).

Safe CUDA wrappers for:
1. [CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
2. [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
3. [cuRAND](https://docs.nvidia.com/cuda/curand/index.html)
4. [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
5. [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
6. [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
7. [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html)
8. [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/)
9. [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/)
10. [cuFILE](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#introduction)

**Contributions welcome!**

# API 👀

It's easy to create a new device and transfer data to the gpu:

```rust
// Get a stream for GPU 0
let ctx = cudarc::driver::CudaContext::new(0)?;
let stream = ctx.default_stream();

// copy a rust slice to the device
let inp = stream.memcpy_stod(&[1.0f32; 100])?;

// or allocate directly
let mut out = stream.alloc_zeros::<f32>(100)?;
```

You can also use the nvrtc api to compile kernels at runtime:

```rust
let ptx = cudarc::nvrtc::compile_ptx("
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}")?;

// Dynamically load it into the device
let module = ctx.load_module(ptx)?;
let sin_kernel = module.load_function("sin_kernel")?;
```

`cudarc` provides a very simple interface to launch kernels using a builder pattern to specify kernel arguments:

```rust
let mut builder = stream.launch_builder(&sin_kernel);
builder.arg(&mut out);
builder.arg(&inp);
builder.arg(&100usize);
unsafe { builder.launch(LaunchConfig::for_num_elems(100)) }?;
```

And of course it's easy to copy things back to host after you're done:

```rust
let out_host: Vec<f32> = stream.memcpy_dtov(&out)?;
assert_eq!(out_host, [1.0; 100].map(f32::sin));
```

# Design

Goals are:
1. As safe as possible (there will still be a lot of unsafe due to ffi & async)
2. As ergonomic as possible
3. Allow mixing of high level `safe` apis, with low level `sys` apis

To that end there are three levels to each wrapper (by default the safe api is exported):
```rust
use cudarc::driver::{safe, result, sys};
use cudarc::nvrtc::{safe, result, sys};
use cudarc::cublas::{safe, result, sys};
use cudarc::cublaslt::{safe, result, sys};
use cudarc::curand::{safe, result, sys};
use cudarc::nccl::{safe, result, sys};
```

where:
1. `sys` is the raw ffi apis generated with bindgen
2. `result` is a very small wrapper around sys to return `Result` from each function
3. `safe` is a wrapper around result/sys to provide safe abstractions

*Heavily recommend sticking with safe APIs*

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
