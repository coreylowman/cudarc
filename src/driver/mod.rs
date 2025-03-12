//! Wrappers around the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html),
//! in three levels. See crate documentation for description of each.
//!
//! # safe api usage
//!
//! 1. Instantiate a [CudaContext]:
//!
//! ```rust
//! # use cudarc::driver::*;
//! let ctx = CudaContext::new(0).unwrap();
//! ```
//!
//! 2. Create a [CudaStream] to schedule work on using [CudaContext::default_stream()] or [CudaContext::new_stream()]:
//!
//! ```rust
//! # use cudarc::driver::*;
//! # let ctx = CudaContext::new(0).unwrap();
//! let stream = ctx.default_stream();
//! ```
//!
//! 3. Allocate device memory with [CudaStream::memcpy_stod()]/[CudaStream::memcpy_htod], [CudaStream::alloc_zeros()].
//!
//! ```rust
//! # use cudarc::driver::*;
//! # let ctx = CudaContext::new(0).unwrap();
//! # let stream = ctx.default_stream();
//! let a_dev: CudaSlice<f32> = stream.alloc_zeros(10).unwrap();
//! let b_dev: CudaSlice<f32> = stream.memcpy_stod(&[0.0; 10]).unwrap();
//! ```
//!
//! 3. Transfer to host memory with [CudaStream::memcpy_dtov()], or [CudaStream::memcpy_dtoh()]
//!
//! ```rust
//! # use cudarc::driver::*;
//! # let ctx = CudaContext::new(0).unwrap();
//! # let stream = ctx.default_stream();
//! let a_dev: CudaSlice<f32> = stream.alloc_zeros(10).unwrap();
//! let mut a_host: [f32; 10] = [1.0; 10];
//! stream.memcpy_dtoh(&a_dev, &mut a_host);
//! assert_eq!(a_host, [0.0; 10]);
//! let a_host: Vec<f32> = stream.memcpy_dtov(&a_dev).unwrap();
//! assert_eq!(&a_host, &[0.0; 10]);
//! ```
//!
//! ## Mutating device memory - [CudaModule]/[CudaFunction]
//!
//! See [CudaStream::launch_builder()]/[LaunchArgs::launch()] and [CudaFunction].
//!
//! In order to mutate device data, you need to use cuda kernels.
//!
//! Loading kernels is done with [CudaContext::load_module()]
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # use std::sync::Arc;
//! let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! let ctx = CudaContext::new(0).unwrap();
//! let module: Arc<CudaModule> = ctx.load_module(ptx).unwrap();
//! ```
//!
//! Retrieve functions using the [CudaModule::load_function()]
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let ctx = CudaContext::new(0).unwrap();
//! # let module = ctx.load_module(ptx).unwrap();
//! let f: CudaFunction = module.load_function("my_function").unwrap();
//! ```
//!
//! Asynchronously execute the kernel:
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let ctx = CudaContext::new(0).unwrap();
//! # let module = ctx.load_module(ptx).unwrap();
//! # let f: CudaFunction = module.load_function("my_function").unwrap();
//! let stream = ctx.default_stream();
//! let mut a = stream.alloc_zeros::<f32>(10).unwrap();
//! let cfg = LaunchConfig::for_num_elems(10);
//! unsafe { stream.launch_builder(&f).arg(&mut a).launch(cfg) }.unwrap();
//! ```
//!
//! Note: Launching kernels is **extremely unsafe**. See [LaunchArgs::launch()] for more info.
//!
//! ## Sub slices of [CudaSlice] - [CudaView] & [CudaViewMut]
//!
//! For some operations, it is necessary to only operate on a small part of a single [CudaSlice].
//! For example, the slice may represent a batch of items, and you want to run separate kernels
//! on each of the items in the batch.
//!
//! Use [CudaSlice::try_slice()] and [CudaSlice::try_slice_mut()] for this. The returned
//! views ([CudaView] and [CudaViewMut] hold references to the owning [CudaSlice],
//! so rust's ownership system handles safety here.
//!
//! These view structs can be used with [CudaFunction], and any [CudaStream] methods.
//!
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let ctx = CudaContext::new(0).unwrap();
//! # let stream = ctx.default_stream();
//! # let module = ctx.load_module(ptx).unwrap();
//! # let f = module.load_function("my_function").unwrap();
//! let cfg = LaunchConfig::for_num_elems(10);
//! let mut a: CudaSlice<f32> = stream.alloc_zeros::<f32>(3 * 10).unwrap();
//! for i_batch in 0..3 {
//!     let mut a_sub_view: CudaViewMut<f32> = a.try_slice_mut(i_batch * 10..).unwrap();
//!     unsafe { stream.launch_builder(&f).arg(&mut a_sub_view).launch(cfg) }.unwrap();
//! }
//! ```
//!
//! #### A note on implementation
//!
//! It would be possible to re-use [CudaSlice] itself for sub-slices, however that would involve adding
//! another structure underneath the hood that is wrapped in an [std::sync::Arc] to minimize data cloning.
//! Overall it seemed more complex than the current implementation.
//!
//! # Multi threading
//!
//! We implement [Send]/[Sync] on all types that it is safe to do so on. [CudaContext] will auto bind to whatever
//! thread is currently using it.
//!
//! # Safety
//!
//! There are a number of aspects to this, but at a high level this API utilizes [std::sync::Arc] to
//! control when [CudaContext] can be dropped.
//!
//! ### Context/Stream lifetimes
//!
//! The first part of safety is ensuring that [crate::driver::sys::CUcontext],
//! [crate::driver::sys::CUdevice], and [crate::driver::sys::CUstream] all
//! live the required amount of time (i.e. device outlives context, which outlives stream).
//!
//! This is accomplished by putting all of them inside one struct, the [CudaContext]. There are other ways,
//! such as adding newtypes that carry lifetimes with them, but this approach was chosen to make working
//! with device pointers easier.
//!
//! Additionally, [CudaContext] implements [Drop] as releasing all the data from the device in
//! the expected way.
//!
//! ### Device Data lifetimes
//!
//! The next part of safety is ensuring that [CudaSlice] do not outlive
//! the [CudaContext]. For usability, each [CudaSlice] owns an `Arc<CudaContext>`
//! to ensure the device stays alive.
//!
//! Additionally we don't want to double free any device pointers, so free is only
//! called when the device pointer is dropped. Thanks rust!
//!
//! ### Host and Device Data lifetimes
//!
//! When copying data between host & device, we ensure proper use of [CudaEvent::synchronize()]
//! and [CudaStream::synchronize()] to make sure no data is freed during use.

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
