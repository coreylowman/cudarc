//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaDevice], [CudaDeviceBuilder], and more.
//!
//! # Usage
//!
//! 1. Instantiate a [CudaDevice] with [CudaDeviceBuilder]:
//!
//! ```rust
//! # use cudarc::prelude::*;
//! let device = CudaDeviceBuilder::new(0).build().unwrap();
//! ```
//!
//! 2. Allocate device memory with host data with [CudaDevice::take_async()], [CudaDevice::alloc_zeros_async()],
//! or [CudaDevice::sync_copy()]
//!
//! You can also copy data to CudaSlice using [CudaDevice::sync_copy_into()]
//!
//! ```rust
//! # use cudarc::prelude::*;
//! # let device = CudaDeviceBuilder::new(0).build().unwrap();
//! let a_dev: CudaSlice<f32> = device.alloc_zeros_async(10).unwrap();
//! let b_dev: CudaSlice<f32> = device.take_async(vec![0.0; 10]).unwrap();
//! let c_dev: CudaSlice<f32> = device.sync_copy(&[1.0, 2.0, 3.0]).unwrap();
//! ```
//!
//! 3. Transfer to host memory with [CudaDevice::sync_release()] or [CudaDevice::sync_copy_from()]
//!
//! ```rust
//! # use cudarc::prelude::*;
//! # use std::rc::Rc;
//! # let device = CudaDeviceBuilder::new(0).build().unwrap();
//! let a_dev: CudaSlice<f32> = device.alloc_zeros_async(10).unwrap();
//! let mut a_buf: [f32; 10] = [1.0; 10];
//! device.sync_copy_from(&a_dev, &mut a_buf);
//! assert_eq!(a_buf, [0.0; 10]);
//! let a_host: Vec<f32> = device.sync_release(a_dev).unwrap();
//! assert_eq!(&a_host, &[0.0; 10]);
//! ```
//!
//! ## Mutating device memory - [CudaFunction]
//!
//! See [LaunchAsync] and [CudaFunction].
//!
//! In order to mutate device data, you need to use cuda kernels.
//!
//! Loading kernels is done with [CudaDeviceBuilder::with_ptx()]
//! and [CudaDeviceBuilder::with_ptx_from_file()]:
//! ```rust
//! # use cudarc::device::*;
//! # use cudarc::jit::*;
//! let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! let device = CudaDeviceBuilder::new(0)
//!     .with_ptx(ptx, "module_name", &["my_function"])
//!     .build()
//!     .unwrap();
//! ```
//!
//! Retrieve the function using the registered module name & actual function name:
//! ```rust
//! # use cudarc::device::*;
//! # use cudarc::jit::*;
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDeviceBuilder::new(0).with_ptx(ptx, "module_name", &["my_function"]).build().unwrap();
//! let func: CudaFunction = device.get_func("module_name", "my_function").unwrap();
//! ```
//!
//! Asynchronously execute the kernel:
//! ```rust
//! # use cudarc::device::*;
//! # use cudarc::jit::*;
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDeviceBuilder::new(0).with_ptx(ptx, "module_key", &["my_function"]).build().unwrap();
//! # let func: CudaFunction = device.get_func("module_key", "my_function").unwrap();
//! let mut a = device.alloc_zeros_async::<f32>(10).unwrap();
//! let cfg = LaunchConfig::for_num_elems(10);
//! unsafe { func.launch_async(cfg, (&mut a,)) }.unwrap();
//! ```
//!
//! Note: Launching kernels is **extremely unsafe**. See [LaunchAsync] for more info.
//!
//! # Safety
//!
//! There are a number of aspects to this, but at a high level this API utilizes [std::sync::Arc] to
//! control when [CudaDevice] can be dropped.
//!
//! ### Context/Stream lifetimes
//!
//! The first part of safety is ensuring that [sys::CUcontext], [sys::CUdevice], and [sys::CUstream] all
//! live the required amount of time (i.e. device outlives context, which outlives stream).
//!
//! This is accomplished by putting all of them inside one struct, the [CudaDevice]. There are other ways,
//! such as adding newtypes that carry lifetimes with them, but this approach was chosen to make working
//! with device pointers easier.
//!
//! Additionally, [CudaDevice] implements [Drop] as releasing all the data from the device in
//! the expected way.
//!
//! ### Device Data lifetimes
//!
//! The next part of safety is ensuring that [CudaSlice] do not outlive
//! the [CudaDevice]. For usability, each [CudaSlice] owns an [`Arc<CudaDevice>`]
//! to ensure the device stays alive.
//!
//! Additionally we don't want to double free any device pointers, so free is only
//! called when the device pointer is dropped. Thanks rust!
//!
//! ### Host and Device Data lifetimes
//!
//! Each device allocation can be associated with a host allocation. We want to ensure
//! that these have the same lifetimes *when copying data between them*.
//!
//! This is done via the various copy methods. Methods that don't take ownership
//! of the host data need to be executed synchronously, while the methods own the reference.
//! Methods that do own the host data can be executed synchronously.
//!
//! ### Single stream operations
//!
//! The next part of safety is ensuring that:
//! 1. The null stream is not used
//! 2. Data isnt mutated by more than 1 stream at a time.
//!
//! At the moment, only a single stream is supported, and only the `*_async` methods
//! in [crate::driver::result] are used.
//!
//! Another important aspect of this is ensuring that mutability in an async setting
//! is sound, and something can't be freed while it's being used in a kernel.

use crate::driver::{result, sys};
use crate::jit::Ptx;
use alloc::ffi::{CString, NulError};
use std::{
    collections::BTreeMap,
    marker::Unpin,
    pin::Pin,
    sync::{Arc, RwLock},
    vec::Vec,
};

pub use result::DriverError;

/// Contains a reference counted pointer to both
/// device and host memory allocated for type `T`.
///
/// # Host data
///
/// *This owns the host data it is associated with*. However
/// it is possible to create device memory without having
/// a corresponding host memory, so the host memory is
/// actually [Option].
///
/// # Reclaiming host data
///
/// To reclaim the host data for this device data,
/// use [CudaDevice::sync_release()]. This will
/// perform necessary synchronization to ensure
/// that the device data finishes copying over.
///
/// # Mutating device data
///
/// This can only be done by launching kernels via
/// [LaunchAsync] which is implemented
/// by [CudaDevice]. Pass `&mut CudaSlice<T>`
/// if you want to mutate the rc, and `&CudaSlice<T>` otherwise.
///
/// Unfortunately, `&CudaSlice<T>` can **still be mutated
/// by the [CudaFunction]**.
#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) host_buf: Option<Pin<Vec<T>>>,
}

unsafe impl<T: Send> Send for CudaSlice<T> {}
unsafe impl<T: Sync> Sync for CudaSlice<T> {}

impl<T> CudaSlice<T> {
    /// Number of elements in the slice
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of the slice in bytes
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn clone_async(&self) -> Result<Self, DriverError> {
        let dst = unsafe { self.device.alloc(self.len) }?;
        unsafe {
            result::memcpy_dtod_async(
                dst.cu_device_ptr,
                self.cu_device_ptr,
                self.num_bytes(),
                self.device.cu_stream,
            )
        }?;
        Ok(dst)
    }
}

impl<T> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.clone_async().unwrap()
    }
}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        unsafe { result::free_async(self.cu_device_ptr, self.device.cu_stream) }.unwrap();
    }
}

impl<T: Clone + Default + Unpin> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.device.clone().sync_release(value)
    }
}

/// A wrapper around [sys::CUdevice], [sys::CUcontext], [sys::CUstream],
/// and [CudaFunction].
///
/// **Must be created through [CudaDeviceBuilder].**
///
/// # Safety
/// 1. impl [Drop] to call all the corresponding resource cleanup methods
/// 2. Doesn't impl clone, so you can't have multiple device pointers
/// hanging around.
/// 3. Any allocations enforce that self is an [Arc], meaning no allocation
/// can outlive the [CudaDevice]
#[derive(Debug)]
pub struct CudaDevice {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_primary_ctx: sys::CUcontext,
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) modules: RwLock<BTreeMap<&'static str, CudaModule>>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        let modules = RwLock::get_mut(&mut self.modules).unwrap();
        for (_, module) in modules.iter() {
            unsafe { result::module::unload(module.cu_module) }.unwrap();
        }
        modules.clear();

        let stream = std::mem::replace(&mut self.cu_stream, std::ptr::null_mut());
        if !stream.is_null() {
            unsafe { result::stream::destroy(stream) }.unwrap();
        }

        let ctx = std::mem::replace(&mut self.cu_primary_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            unsafe { result::primary_ctx::release(self.cu_device) }.unwrap();
        }
    }
}

impl CudaDevice {
    /// Allocates device memory and increments the reference counter of [CudaDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    unsafe fn alloc<T>(self: &Arc<Self>, len: usize) -> Result<CudaSlice<T>, DriverError> {
        let cu_device_ptr = result::malloc_async(self.cu_stream, len * std::mem::size_of::<T>())?;
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
        })
    }

    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn alloc_zeros_async<T: ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let dst = unsafe { self.alloc(len) }?;
        unsafe { result::memset_d8_async(dst.cu_device_ptr, 0, dst.num_bytes(), self.cu_stream) }?;
        Ok(dst)
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn take_async<T: Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.copy_into_async(src, &mut dst)?;
        Ok(dst)
    }

    /// Allocates new device memory and synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::take_async()].
    ///
    /// # Safety
    ///
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy<T>(self: &Arc<Self>, src: &[T]) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.sync_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::take_async()].
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy_into<T>(
        self: &Arc<Self>,
        src: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe { result::memcpy_htod_async(dst.cu_device_ptr, src, self.cu_stream) }?;
        self.synchronize()
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_into_async<T: Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), DriverError> {
        assert_eq!(src.len(), dst.len());
        dst.host_buf = Some(Pin::new(src));
        unsafe {
            result::memcpy_htod_async(
                dst.cu_device_ptr,
                dst.host_buf.as_ref().unwrap(),
                self.cu_stream,
            )
        }?;
        Ok(())
    }

    /// Synchronously copies device memory into host memory
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy_from<T>(
        self: &Arc<Self>,
        src: &CudaSlice<T>,
        dst: &mut [T],
    ) -> Result<(), DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe { result::memcpy_dtoh_async(dst, src.cu_device_ptr, self.cu_stream) }?;
        self.synchronize()
    }

    /// De-allocates `src` and converts it into it's host value. You can just drop the slice if you don't
    /// need the host data.
    ///
    /// # Safety
    /// 1. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_release<T: Clone + Default + Unpin>(
        self: &Arc<Self>,
        mut src: CudaSlice<T>,
    ) -> Result<Vec<T>, DriverError> {
        let buf = src.host_buf.take();
        let mut buf = buf.unwrap_or_else(|| Pin::new(std::vec![Default::default(); src.len]));
        self.sync_copy_from(&src, &mut buf)?;
        Ok(Pin::into_inner(buf))
    }

    /// Synchronizes the stream.
    pub fn synchronize(self: &Arc<Self>) -> Result<(), DriverError> {
        unsafe { result::stream::synchronize(self.cu_stream) }
    }

    /// Whether a module and function are currently loaded into the device.
    pub fn has_func(self: &Arc<Self>, module_name: &str, func_name: &str) -> bool {
        let modules = self.modules.read().unwrap();
        modules
            .get(module_name)
            .map_or(false, |module| module.has_func(func_name))
    }

    /// Retrieves a [CudaFunction] that was registered under `module_name` and `func_name`.
    pub fn get_func(self: &Arc<Self>, module_name: &str, func_name: &str) -> Option<CudaFunction> {
        let modules = self.modules.read().unwrap();
        modules
            .get(module_name)
            .and_then(|m| m.get_func(func_name))
            .map(|cu_function| CudaFunction {
                cu_function,
                device: self.clone(),
            })
    }

    /// Dynamically load a set of [CudaFunction] from a ptx file. See [CudaDeviceBuilder::with_ptx_from_file].
    pub fn load_ptx_from_file(
        self: &Arc<Self>,
        ptx_path: &'static str,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), BuildError> {
        let m = CudaDeviceBuilder::build_module_from_ptx_file(ptx_path, module_name, func_names)?;
        {
            let mut modules = self.modules.write().unwrap();
            modules.insert(module_name, m);
        }
        Ok(())
    }

    /// Dynamically load a set of [CudaFunction] from a jit compiled ptx. See [CudaDeviceBuilder::with_ptx]
    pub fn load_ptx(
        self: &Arc<Self>,
        ptx: Ptx,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), BuildError> {
        let m = CudaDeviceBuilder::build_module_from_ptx(ptx, module_name, func_names)?;
        {
            let mut modules = self.modules.write().unwrap();
            modules.insert(module_name, m);
        }
        Ok(())
    }
}

/// Wrapper around [sys::CUmodule] that also contains
/// the loaded [CudaFunction] associated with this module.
///
/// See [CudaModule::get_fn()] for retrieving function handles.
///
/// See [CudaDeviceBuilder] for how to construct these modules.
#[derive(Debug)]
pub(crate) struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: BTreeMap<&'static str, sys::CUfunction>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Returns reference to function with `name`. If function
    /// was not already loaded into CudaModule, then `None`
    /// is returned.
    pub(crate) fn get_func(&self, name: &str) -> Option<sys::CUfunction> {
        self.functions.get(name).cloned()
    }

    pub(crate) fn has_func(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

/// Wrapper around [sys::CUfunction]. Used by [LaunchAsync].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    pub(crate) device: Arc<CudaDevice>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

/// Configuration for [result::launch_kernel]
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
/// for description of each parameter.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// A simple function to create launch configuration
    /// with 1 grid and n threads
    pub fn for_num_elems(n: u32) -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (n, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Something that can be turned into a parameter for
/// [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because it can take any type and
/// turn it into a mutable pointer.
///
/// Additionally, all the safety notices for [result::launch_kernel]
/// apply here as well.
pub unsafe trait IntoKernelParam {
    fn into_kernel_param(self) -> *mut std::ffi::c_void;
}

/// Consumes a [CudaFunction] to execute asychronously on the device with
/// params determined by generic parameter `Params`.
///
/// This is impl'd multiple times for different number and types of params. In
/// general, `Params` should impl [IntoKernelParam]
///
/// # Safety
///
/// This is not safe really ever, because there's no garuntee that `Params`
/// will work for any [CudaFunction] passed in. Great care should be taken
/// to ensure that [CudaFunction] works with `Params` and that the correct
/// parameters have `&mut` in front of them.
///
/// Additionally, kernels can mutate data that is marked as immutable,
/// such as `&CudaSlice<T>`.
///
/// See [LaunchAsync::launch_async] for more details
pub unsafe trait LaunchAsync<Params> {
    /// Launches the [CudaFunction] with the corresponding `Params`.
    ///
    /// # Safety
    ///
    /// This method is **very** unsafe.
    ///
    /// See cuda documentation notes on this as well:
    /// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#functions>
    ///
    /// 1. `params` can be changed regardless of `&` or `&mut` usage.
    /// 2. `params` will be changed at some later point after the
    /// function returns because the kernel is executed async.
    /// 3. There are no guaruntees that the `params`
    /// are the correct number/types/order for `func`.
    /// 4. Specifying the wrong values for [LaunchConfig] can result
    /// in accessing/modifying values past memory limits.
    ///
    /// ## Asynchronous mutation
    ///
    /// Since this library queues kernels to be launched on a single
    /// stream, and really the only way to modify [CudaSlice] is through
    /// kernels, mutating the same [CudaSlice] with multiple kernels
    /// is safe. This is because each kernel is executed sequentially
    /// on the stream.
    ///
    /// **Modifying a value on the host that is in used by a
    /// kernel is undefined behavior.** But is hard to do
    /// accidentally.
    ///
    /// Also for this reason, do not pass in any values to kernels
    /// that can be modified on the host. This is the reason
    /// [IntoKernelParam] is not implemented for rust primitive
    /// references.
    ///
    /// ## Use after free
    ///
    /// Since the drop implementation for [CudaSlice] also occurs
    /// on the device's single stream, any kernels launched before
    /// the drop will complete before the value is actually freed.
    ///
    /// **If you launch a kernel or drop a value on a different stream
    /// this may not hold**
    unsafe fn launch_async(self, cfg: LaunchConfig, params: Params) -> Result<(), DriverError>;
}

unsafe impl<T> IntoKernelParam for &mut CudaSlice<T> {
    #[inline(always)]
    fn into_kernel_param(self) -> *mut std::ffi::c_void {
        (&mut self.cu_device_ptr) as *mut sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<T> IntoKernelParam for &CudaSlice<T> {
    #[inline(always)]
    fn into_kernel_param(self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

macro_rules! impl_into_kernel_param {
    ($T:ty) => {
        unsafe impl IntoKernelParam for $T {
            #[inline(always)]
            fn into_kernel_param(self) -> *mut std::ffi::c_void {
                (&self) as *const $T as *mut std::ffi::c_void
            }
        }
    };
}

impl_into_kernel_param!(i8);
impl_into_kernel_param!(i16);
impl_into_kernel_param!(i32);
impl_into_kernel_param!(i64);
impl_into_kernel_param!(isize);
impl_into_kernel_param!(u8);
impl_into_kernel_param!(u16);
impl_into_kernel_param!(u32);
impl_into_kernel_param!(u64);
impl_into_kernel_param!(usize);
impl_into_kernel_param!(f32);
impl_into_kernel_param!(f64);

macro_rules! impl_launch {
    ([$($Vars:tt),*], [$($Idx:tt),*]) => {
unsafe impl<$($Vars: IntoKernelParam),*> LaunchAsync<($($Vars, )*)> for CudaFunction {
    unsafe fn launch_async(
        self,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), DriverError> {
        let params = &mut [$(args.$Idx.into_kernel_param(), )*];
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.device.cu_stream,
            params,
        )
    }
}
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);

/// A builder for [CudaDevice].
///
/// Call [CudaDeviceBuilder::new()] to start, and [CudaDeviceBuilder::build]
/// to finish.
///
/// Provides a way to specify what modules & functions to load into
/// the device via [CudaDeviceBuilder::with_ptx_from_file()]
/// and [CudaDeviceBuilder::with_ptx()].
#[derive(Debug)]
pub struct CudaDeviceBuilder {
    pub(crate) ordinal: usize,
    pub(crate) ptx_files: Vec<PtxFileConfig>,
    pub(crate) ptxs: Vec<PtxConfig>,
}

#[derive(Debug)]
pub(crate) struct PtxFileConfig {
    pub(crate) key: &'static str,
    pub(crate) fname: &'static str,
    pub(crate) fn_names: Vec<&'static str>,
}

#[derive(Debug)]
pub(crate) struct PtxConfig {
    pub(crate) key: &'static str,
    pub(crate) ptx: Ptx,
    pub(crate) fn_names: Vec<&'static str>,
}

impl CudaDeviceBuilder {
    /// Starts a new builder object.
    /// - `ordinal` is the index of th cuda device to attach to.
    pub fn new(ordinal: usize) -> Self {
        Self {
            ordinal,
            ptx_files: Vec::new(),
            ptxs: Vec::new(),
        }
    }

    /// Adds a path to a precompiled `.ptx` file to be loaded as a module on the device.
    ///
    /// - `ptx_path` is a file
    /// - `key` is a unique identifier used to access the module later on with [CudaDevice::get_func()]
    /// - `fn_names` is a slice of function names to load into the module during build.
    pub fn with_ptx_from_file(
        mut self,
        ptx_path: &'static str,
        key: &'static str,
        fn_names: &[&'static str],
    ) -> Self {
        self.ptx_files.push(PtxFileConfig {
            key,
            fname: ptx_path,
            fn_names: fn_names.to_vec(),
        });
        self
    }

    /// Add a [Ptx] compiled with nvrtc to be loaded as a module on the device.
    ///
    /// - `key` is a unique identifier used to access the module later on with [CudaDevice::get_func()]
    /// - `ptx` contains the compilex ptx
    /// - `fn_names` is a slice of function names to load into the module during build.
    pub fn with_ptx(mut self, ptx: Ptx, key: &'static str, fn_names: &[&'static str]) -> Self {
        self.ptxs.push(PtxConfig {
            key,
            ptx,
            fn_names: fn_names.to_vec(),
        });
        self
    }

    /// Builds the [CudaDevice]:
    /// 1. Initializes cuda with [result::init]
    /// 2. Creates the device/primary ctx, and stream
    /// 3. Uses nvrtc to compile and the modules & functions
    pub fn build(mut self) -> Result<Arc<CudaDevice>, BuildError> {
        result::init().map_err(BuildError::InitError)?;

        let cu_device =
            result::device::get(self.ordinal as i32).map_err(BuildError::DeviceError)?;

        // primary context initialization
        let cu_primary_ctx =
            unsafe { result::primary_ctx::retain(cu_device) }.map_err(BuildError::ContextError)?;

        unsafe { result::ctx::set_current(cu_primary_ctx) }.map_err(BuildError::ContextError)?;

        // stream initialization
        let cu_stream = result::stream::create(result::stream::StreamKind::NonBlocking)
            .map_err(BuildError::StreamError)?;

        let mut modules = BTreeMap::new();

        for cu in self.ptx_files.drain(..) {
            modules.insert(
                cu.key,
                Self::build_module_from_ptx_file(cu.fname, cu.key, &cu.fn_names)?,
            );
        }

        for ptx in self.ptxs.drain(..) {
            modules.insert(
                ptx.key,
                Self::build_module_from_ptx(ptx.ptx, ptx.key, &ptx.fn_names)?,
            );
        }

        let device = CudaDevice {
            cu_device,
            cu_primary_ctx,
            cu_stream,
            modules: RwLock::new(modules),
        };
        Ok(Arc::new(device))
    }

    fn build_module_from_ptx_file(
        ptx_path: &'static str,
        key: &'static str,
        func_names: &[&'static str],
    ) -> Result<CudaModule, BuildError> {
        let name_c = CString::new(ptx_path).map_err(BuildError::CStringError)?;
        let cu_module = result::module::load(name_c)
            .map_err(|cuda| BuildError::PtxLoadingError { key, cuda })?;
        let mut functions = BTreeMap::new();
        for &fn_name in func_names.iter() {
            let fn_name_c = CString::new(fn_name).map_err(BuildError::CStringError)?;
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }
                .map_err(|e| BuildError::GetFunctionError {
                    key,
                    symbol: fn_name,
                    cuda: e,
                })?;
            functions.insert(fn_name, cu_function);
        }
        Ok(CudaModule {
            cu_module,
            functions,
        })
    }

    fn build_module_from_ptx(
        ptx: Ptx,
        key: &'static str,
        fn_names: &[&'static str],
    ) -> Result<CudaModule, BuildError> {
        let image = ptx.image.as_ptr() as *const _;
        let cu_module = unsafe { result::module::load_data(image) }
            .map_err(|cuda| BuildError::NvrtcLoadingError { key, cuda })?;
        let mut functions = BTreeMap::new();
        for &fn_name in fn_names.iter() {
            let fn_name_c = CString::new(fn_name).map_err(BuildError::CStringError)?;
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }
                .map_err(|e| BuildError::GetFunctionError {
                    key,
                    symbol: fn_name,
                    cuda: e,
                })?;
            functions.insert(fn_name, cu_function);
        }
        Ok(CudaModule {
            cu_module,
            functions,
        })
    }
}

/// An error the occurs during [CudaDeviceBuilder::build]
#[derive(Debug)]
pub enum BuildError {
    InitError(DriverError),
    DeviceError(DriverError),
    ContextError(DriverError),
    StreamError(DriverError),
    PtxLoadingError {
        key: &'static str,
        cuda: DriverError,
    },
    NvrtcLoadingError {
        key: &'static str,
        cuda: DriverError,
    },
    GetFunctionError {
        key: &'static str,
        symbol: &'static str,
        cuda: DriverError,
    },
    CStringError(NulError),
}

#[cfg(feature = "std")]
impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {}

/// Marker trait to indicate that the type is valid
/// when all of its bits are set to 0.
///
/// # Safety
/// Not all types are valid when all bits are set to 0.
/// Be very sure when implementing this trait!
pub unsafe trait ValidAsZeroBits {}
unsafe impl ValidAsZeroBits for i8 {}
unsafe impl ValidAsZeroBits for i16 {}
unsafe impl ValidAsZeroBits for i32 {}
unsafe impl ValidAsZeroBits for i64 {}
unsafe impl ValidAsZeroBits for isize {}
unsafe impl ValidAsZeroBits for u8 {}
unsafe impl ValidAsZeroBits for u16 {}
unsafe impl ValidAsZeroBits for u32 {}
unsafe impl ValidAsZeroBits for u64 {}
unsafe impl ValidAsZeroBits for usize {}
unsafe impl ValidAsZeroBits for f32 {}
unsafe impl ValidAsZeroBits for f64 {}
unsafe impl<T: ValidAsZeroBits, const M: usize> ValidAsZeroBits for [T; M] {}

#[cfg(test)]
mod tests {
    use crate::jit::compile_ptx_with_opts;

    use super::*;

    #[test]
    fn test_post_build_arc_count() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.alloc_zeros_async::<f32>(1).unwrap();
        assert!(t.host_buf.is_none());
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    fn test_post_take_arc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async(std::vec![0.0f32; 5]).unwrap();
        assert!(t.host_buf.is_some());
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async(std::vec![0.0f64; 10]).unwrap();
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_arc_slice_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = Arc::new(device.take_async(std::vec![0.0f64; 10]).unwrap());
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_release_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async(std::vec![1.0f32, 2.0, 3.0]).unwrap();
        #[allow(clippy::redundant_clone)]
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);

        let r_host = device.sync_release(r).unwrap();
        assert_eq!(&r_host, &[1.0, 2.0, 3.0]);
        assert_eq!(Arc::strong_count(&device), 2);

        drop(r_host);
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let (free1, total1) = result::mem_get_info().unwrap();

        let t = device.take_async(std::vec![0.0f32; 5]).unwrap();
        let (free2, total2) = result::mem_get_info().unwrap();
        assert_eq!(total1, total2);
        assert!(free2 < free1);

        drop(t);
        device.synchronize().unwrap();

        let (free3, total3) = result::mem_get_info().unwrap();
        assert_eq!(total2, total3);
        assert!(free3 > free2);
        assert_eq!(free3, free1);
    }

    #[test]
    fn test_mut_into_kernel_param_no_inc_rc() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let mut t = device.take_async(std::vec![0.0f32; 1]).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&mut t).into_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    #[test]
    fn test_ref_into_kernel_param_inc_rc() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async(std::vec![0.0f32; 1]).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).into_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    const SIN_CU: &str = "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}";

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "sin", &["sin_kernel"])
            .build()
            .unwrap();
        let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();

        let a_host = std::vec![-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];

        let a_dev = dev.take_async(a_host.clone()).unwrap();

        let mut b_dev = a_dev.clone();

        unsafe {
            sin_kernel.launch_async(
                LaunchConfig::for_num_elems(10),
                (&mut b_dev, &a_dev, 10usize),
            )
        }
        .unwrap();

        let b_host = dev.sync_release(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    const TEST_KERNELS: &str = "
extern \"C\" __global__ void int_8bit(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    assert(s_min == -128);
    assert(s_max == 127);
    assert(u_min == 0);
    assert(u_max == 255);
}

extern \"C\" __global__ void int_16bit(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    assert(s_min == -32768);
    assert(s_max == 32767);
    assert(u_min == 0);
    assert(u_max == 65535);
}

extern \"C\" __global__ void int_32bit(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    assert(s_min == -2147483648);
    assert(s_max == 2147483647);
    assert(u_min == 0);
    assert(u_max == 4294967295);
}

extern \"C\" __global__ void int_64bit(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    assert(s_min == -9223372036854775808);
    assert(s_max == 9223372036854775807);
    assert(u_min == 0);
    assert(u_max == 18446744073709551615);
}

extern \"C\" __global__ void floating(float f, double d) {
    printf(\"%.10f %.20f\", f, d);
    assert(fabs(f - 1.2345678) <= 1e-7);
    assert(fabs(d - -10.123456789876543) <= 1e-16);
}
";

    #[test]
    fn test_launch_with_8bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_8bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_8bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i8::MIN, i8::MAX, u8::MIN, u8::MAX),
            )
        }
        .unwrap();

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_16bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_16bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_16bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i16::MIN, i16::MAX, u16::MIN, u16::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_32bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_32bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_32bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i32::MIN, i32::MAX, u32::MIN, u32::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_64bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_64bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_64bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i64::MIN, i64::MAX, u64::MIN, u64::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_floats() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["floating"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "floating").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (1.2345678f32, -10.123456789876543f64),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }
}
