# cudarc: minimal and safe api over the cuda toolkit

[![](https://dcbadge.vercel.app/api/server/AtUhGqBDP5)](https://discord.gg/AtUhGqBDP5)
[![crates.io](https://img.shields.io/crates/v/cudarc?style=for-the-badge)](https://crates.io/crates/cudarc)
[![docs.rs](https://img.shields.io/docsrs/cudarc?label=docs.rs%20latest&style=for-the-badge)](https://docs.rs/cudarc)

Checkout cudarc on [crates.io](https://crates.io/crates/cudarc) and [docs.rs](https://docs.rs/cudarc/latest/cudarc/).

**Contributions welcome!**

Safe CUDA wrappers for:

| library | dynamic load | dynamic link | static link |
| --- | --- | --- | --- |
| [CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) | ✅ | ✅ | ❌ |
| [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) | ✅ | ✅ | ✅ |
| [cuRAND](https://docs.nvidia.com/cuda/curand/index.html) | ✅ | ✅ | ✅ |
| [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) | ✅ | ✅ | ✅ |
| [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api) | ✅ | ✅ | ✅ |
| [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) | ✅ | ✅ | ✅ |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html) | ✅ | ✅ | ✅ |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | ✅ | ✅ | ✅ |
| [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/) | ✅ | ✅ | ❌ |
| [cuFILE](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#introduction) | ✅ | ✅ | ✅ |
| [CUPTI](https://docs.nvidia.com/cupti/) | ✅ | ✅ | ✅ |
| [nvtx](https://nvidia.github.io/NVTX/) | ✅ | ✅ | ❌ |

CUDA Versions supported
- 11.4-11.8
- 12.0-12.9
- 13.0

CUDNN versions supported:
- 9.12.0

NCCL versions supported:
- 2.28.3

# Configuring CUDA version

Select cuda version with one of:
- `-F cuda-version-from-build-system`: At build time will get the cuda toolkit version using `nvcc`
    - `-F fallback-latest`: can be used to control behavior if this fails. default is not enabled, which will cause the build
      script to panic. if `-F fallback-latest` is enabled, we will use the highest bindings we have.
- `-F cuda-<major>0<minor>0` to build for a specific version of cuda

# Configuring linking

By default we use `-F dynamic-loading`, which will not require any libraries to be present at build time.

You can also enable `-F dynamic-linking` or `-F static-linking` for your use case.

# API 👀

It's easy to create a new device and transfer data to the gpu:

```rust
// Get a stream for GPU 0
let ctx = cudarc::driver::CudaContext::new(0)?;
let stream = ctx.default_stream();

// copy a rust slice to the device
let inp = stream.clone_htod(&[1.0f32; 100])?;

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
let out_host: Vec<f32> = stream.clone_dtoh(&out)?;
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
