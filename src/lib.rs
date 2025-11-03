//! Safe CUDA wrappers for:
//! 1. [CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! 2. [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
//! 3. [cuRAND](https://docs.nvidia.com/cuda/curand/index.html)
//! 4. [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
//! 5. [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
//! 6. [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
//! 7. [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html)
//! 8. [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/)
//! 9. [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/)
//! 10. [cuFILE](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#introduction)
//! 11. [CUPTI](https://docs.nvidia.com/cupti/)
//! 12. [nvtx](https://nvidia.github.io/NVTX/)
//! 13. [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/index.html)
//!
//! # crate organization
//!
//! Each of the modules for the above is organized into three levels:
//! 1. A `safe` module which provides safe abstractions over the `result` module
//! 2. A `result` which is a thin wrapper around the `sys` module to ensure all functions return [Result]
//! 3. A `sys` module which contains the raw FFI bindings
//!
//! | API | Safe | Result | Sys |
//! | --- | --- | --- | --- |
//! | driver | [driver::safe] | [driver::result] | [driver::sys] |
//! | cublas | [cublas::safe] | [cublas::result] | [cublas::sys] |
//! | cublaslt | [cublaslt::safe] | [cublaslt::result] | [cublaslt::sys] |
//! | nvrtc | [nvrtc::safe] | [nvrtc::result] | [nvrtc::sys] |
//! | curand | [curand::safe] | [curand::result] | [curand::sys] |
//! | cudnn | [cudnn::safe] | [cudnn::result] | [cudnn::sys] |
//! | cusparse | - | [cusparse::result] | [cusparse::sys] |
//! | cusolver | [cusolver::safe] | [cusolver::result] | [cusolver::sys] |
//! | cusolvermg | [cusolvermg::safe] | [cusolvermg::result] | [cusolvermg::sys] |
//! | cupti | - | [cupti::result] | [cupti::sys] |
//! | nvtx | [nvtx::safe] | [nvtx::result] | [nvtx::sys] |
//! | cutensor | [cutensor::safe] | [cutensor::result] | [cutensor::sys] |
//!
//! # Core Concepts
//!
//! At the core is the [driver] API, which exposes a bunch of structs, but the main ones are:
//!
//! 1. [`driver::CudaContext`] is a handle to a specific device ordinal (e.g. 0, 1, 2, ...)
//! 2. [`driver::CudaStream`] is how you submit work to a device
//! 3. [`driver::CudaSlice<T>`], which represents a [`Vec<T>`] on the device, can be allocated
//!    using the aforementioned [`driver::CudaStream`].
//!
//! Here is a table of similar concepts between CPU and Cuda:
//!
//! | Concept | CPU | Cuda |
//! | --- | --- | --- |
//! | Memory allocator | [`std::alloc::GlobalAlloc`] | [`driver::CudaContext`] |
//! | List of values on heap | [`Vec<T>`] | [`driver::CudaSlice<T>`] |
//! | Slice | `&[T]` | [`driver::CudaView<T>`] |
//! | Mutable Slice  | `&mut [T]` | [`driver::CudaViewMut<T>`] |
//! | Function | [`Fn`] | [`driver::CudaFunction`] |
//! | Calling a function | `my_function(a, b, c)` | [`driver::LaunchArgs::launch()`] |
//! | Thread | [`std::thread::Thread`] | [`driver::CudaStream`] |
//!
//! # Combining the different APIs
//!
//! All the highest level apis have been designed to work together.
//!
//! ## nvrtc
//!
//! [`nvrtc::compile_ptx()`] outputs a [`nvrtc::Ptx`], which can
//! be loaded into a device with [`driver::CudaContext::load_module()`].
//!
//! ## cublas
//!
//! [cublas::CudaBlas] can perform gemm operations using [`cublas::Gemm<T>`],
//! and [`cublas::Gemv<T>`]. Both of these traits can generically accept memory
//! allocated by the driver in the form of: [`driver::CudaSlice<T>`],
//! [`driver::CudaView<T>`], and [`driver::CudaViewMut<T>`].
//!
//! ## curand
//!
//! [curand::CudaRng] can fill a [`driver::CudaSlice<T>`] with random data, based on
//! one of its available distributions.
//!
//! # Combining safe/result/sys
//!
//! The result and sys levels are very inter-changeable for each API. However,
//! the safe apis don't necessarily allow you to mix in the result level. This
//! is to encourage going through the safe API when possible.
//!
//! **If you need some functionality that isn't present in the safe api, please
//! open a ticket.**

#![cfg_attr(feature = "no-std", no_std)]

#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;

#[cfg(feature = "cublas")]
pub mod cublas;
#[cfg(feature = "cublaslt")]
pub mod cublaslt;
#[cfg(feature = "cudnn")]
pub mod cudnn;
#[cfg(feature = "cufile")]
pub mod cufile;
#[cfg(feature = "cupti")]
pub mod cupti;
#[cfg(feature = "curand")]
pub mod curand;
#[cfg(feature = "cusolver")]
pub mod cusolver;
#[cfg(feature = "cusolvermg")]
pub mod cusolvermg;
#[cfg(feature = "cusparse")]
pub mod cusparse;
#[cfg(feature = "cutensor")]
pub mod cutensor;
#[cfg(feature = "driver")]
pub mod driver;
#[cfg(feature = "nccl")]
pub mod nccl;
#[cfg(feature = "nvrtc")]
pub mod nvrtc;
#[cfg(feature = "nvtx")]
pub mod nvtx;
#[cfg(feature = "runtime")]
pub mod runtime;

pub mod types;

#[cfg(feature = "dynamic-loading")]
pub(crate) fn panic_no_lib_found<S: std::fmt::Debug>(lib_name: &str, choices: &[S]) -> ! {
    panic!("Unable to dynamically load the \"{lib_name}\" shared library - searched for library names: {choices:?}. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.");
}

#[cfg(feature = "dynamic-loading")]
pub(crate) fn get_lib_name_candidates(lib_name: &str) -> std::vec::Vec<std::string::String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let pointer_width = if cfg!(target_pointer_width = "32") {
        "32"
    } else if cfg!(target_pointer_width = "64") {
        "64"
    } else {
        panic!("Unsupported target pointer width")
    };

    let major = env!("CUDA_MAJOR_VERSION");
    let minor = env!("CUDA_MINOR_VERSION");

    [
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{minor}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{minor}_0{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}0_{minor}{DLL_SUFFIX}"),
        // See issue #242
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_10{DLL_SUFFIX}"),
        // See issue #246
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}0_0{DLL_SUFFIX}"),
        // See issue #260
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_9{DLL_SUFFIX}"),
        // See issue #274
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.{major}"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.11"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.10"),
        // See issue #296
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.1"),
    ]
    .into()
}
