//! Safe abstractions over [result] provided by [CudaRc], [CudaDevice], [CudaDeviceBuilder], and more.
//!
//! # Usage
//!
//! ### Instanting a [CudaDevice] with [CudaDeviceBuilder]
//!
//! TODO
//!
//! ### Allocating device memory with host data
//!
//! See [CudaDevice::take()]
//!
//! TODO
//!
//! ### Allocating device memory without host data
//!
//! See [CudaDevice::alloc_zeros()]
//!
//! TODO
//!
//! ### Mutating device memory with [LaunchCudaFunction] and [CudaFunction]
//!
//! TODO
//!
//! ### Reclaiming device data
//!
//! See [CudaRc::into_host()], [CudaRc::sync_release()], or you can just
//! drop the [CudaRc].
//!
//! TODO
//!
//! # Safety
//!
//! There are a number of aspects to this, but at a high level this API utilizes [std::rc::Rc] as well
//! as proper management of resources.
//!
//! ### Context/Stream lifetimes
//!
//! The first part of safety is ensuring that [sys::CUcontext], [sys::CUdevice], and [sys::CUstream] all
//! live the required amount of time (i.e. stream requires context, context requires device).
//!
//! This is accomplished by putting all of them inside one struct, the [CudaDevice]. There are other ways,
//! such as adding newtypes that carry lifetimes with them, but this approach was chosen to make working
//! with device pointers easier.
//!
//! Additionally, [CudaDevice] implements [Drop] as releasing all the data from the device in
//! the expected way.
//!
//! ### Single stream operations
//!
//! The next part of safety is ensuring that all operations happen on the same stream. This
//! is pretty easy to accomplish by using all the `*_async` methods in [crate::cuda::result].
//! Otherwise there can be confusing with data copying if you don't use all async methods.
//!
//! ### Device Pointer lifetimes
//!
//! The next part of safety is ensuring that [sys::CUdeviceptr] do not outlive
//! the [CudaDevice]. Again it is possible to do this with lifetimes, but for usability
//! we choose to bundle [Rc<CudaDevice>] along with every [sys::CUdeviceptr].
//!
//! Additionally we don't want to double free any device pointers, so free is only
//! called when the device pointer is dropped. Thanks rust!
//!
//! ### Host and Device Data lifetimes
//!
//! Each device allocation can be associated with a host allocation. We want to ensure
//! that these have the same lifetimes.
//!
//! This is done in [CudaRc<T>], which owns both the (optional) host and device data.
//! In order to initialize device data for a host allocation, you can call
//! [CudaDevice::take()], and to reclaim (& sync) the host data, you can call
//! [CudaRc::into_host()].

use super::result;
use super::sys;
use crate::nvrtc::compile::Ptx;
use std::alloc::alloc_zeroed;
use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::rc::Rc;

pub use result::CudaError;

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
/// # Reference counting
///
/// When cloned it will increment reference counters
/// instead of cloning actual data.
///
/// # Reclaiming host data
///
/// To reclaim the host data for this device data,
/// use [CudaRc::sync_release()] or [CudaRc::into_host()].
/// These will both perform necessary synchronization to ensure
/// that the device data finishes copying over.
///
/// # Mutating device data
///
/// This can only be done by launching kernels via
/// [LaunchCudaFunction] which is implemented
/// by [CudaDevice]. Pass `&mut CudaRc<T>`
/// if you want to mutate the rc, and `&CudaRc<T>` otherwise.
///
/// Unfortunately, `&CudaRc<T>` can **still be mutated
/// by the [CudaFunction]**.
#[derive(Debug, Clone)]
pub struct CudaRc<T> {
    pub(crate) t_cuda: Rc<CudaUniquePtr<T>>,
    pub(crate) t_host: Option<Rc<T>>,
}

impl<T> CudaRc<T> {
    /// Returns a reference to the underlying [CudaDevice]
    pub fn device(&self) -> &Rc<CudaDevice> {
        &self.t_cuda.device
    }
}

impl<T: Clone> CudaRc<T> {
    /// Copies device memory into host memory if it exists,
    /// synchronizes the stream, and then returns the host data.
    ///
    /// Note: This decrements the reference count on the device memory,
    /// so if there no other references to it, it will be freed.
    pub fn sync_release(mut self) -> Result<Option<Rc<T>>, CudaError> {
        self.t_cuda.device.clone().maybe_sync_host(&mut self)?;
        Ok(self.t_host)
    }

    /// If the host data doesn't exist, allocates zerod memory
    /// for the type. Then calls [Self::sync_release] and unwraps
    /// the option.
    ///
    /// Note: This decrements the reference count on the device memory,
    /// so if there no other references to it, it will be freed.
    ///
    /// # Safety
    /// Even though this allocates zerod memory for `T`
    /// and `T` is not necessarily [ValidAsZeroBits], it
    /// is safe since the device memory is valid for T.
    pub fn into_host(mut self) -> Result<Rc<T>, CudaError> {
        self.t_host.get_or_insert_with(|| {
            let layout = Layout::new::<T>();
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                Box::from_raw(ptr).into()
            }
        });
        self.sync_release().map(Option::unwrap)
    }
}

/// Wrapper around [sys::CUdeviceptr] that also contains a [Rc<CudaDevice>].
/// This helps with safety because it:
/// 1. Ensures that the device pointer is associated with the type `T` it was created with
/// 2. Makes the CudaDevice stay alive as long as this object lives
/// 3. impl [Drop] to properly free resources with the device's stream.
/// 4. impl [Clone] as actually doing a device allocation instead of cloning the
/// device pointer.
///
/// This can only be created by [CudaDevice::alloc].
#[derive(Debug)]
pub(crate) struct CudaUniquePtr<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) device: Rc<CudaDevice>,
    marker: PhantomData<*const T>,
}

impl<T> CudaUniquePtr<T> {
    /// Allocates device memory and increments the reference counter to [CudaDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    unsafe fn alloc(device: &Rc<CudaDevice>) -> Result<CudaUniquePtr<T>, CudaError> {
        let cu_device_ptr = result::malloc_async::<T>(device.cu_stream)?;
        Ok(CudaUniquePtr {
            cu_device_ptr,
            device: device.clone(),
            marker: PhantomData,
        })
    }

    /// Allocates new memory for type `T` and schedules a device to device copy of memory.
    fn dup(&self) -> Result<CudaUniquePtr<T>, CudaError> {
        let alloc = unsafe { Self::alloc(&self.device) }?;
        unsafe {
            result::memcpy_dtod_async::<T>(
                alloc.cu_device_ptr,
                self.cu_device_ptr,
                self.device.cu_stream,
            )
        }?;
        Ok(alloc)
    }
}

impl<T> Clone for CudaUniquePtr<T> {
    fn clone(&self) -> Self {
        self.dup().unwrap()
    }
}

impl<T> Drop for CudaUniquePtr<T> {
    fn drop(&mut self) {
        unsafe { result::free_async(self.cu_device_ptr, self.device.cu_stream) }.unwrap();
    }
}

/// A wrapper around [sys::CUdevice], [sys::CUcontext], [sys::CUstream],
/// and [CudaModule]s.
///
/// **Must be created through [CudaDeviceBuilder].**
///
/// # Safety
/// 1. impl [Drop] to call all the corresponding resource cleanup methods
/// 2. Doesn't impl clone, so you can't have multiple device pointers
/// hanging around.
/// 3. Any allocations enforce that self is an [Rc], meaning no allocation
/// can outlive the [CudaDevice]
#[derive(Debug)]
pub struct CudaDevice {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_primary_ctx: sys::CUcontext,
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) modules: HashMap<&'static str, CudaModule>,
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        for (_, module) in self.modules.drain() {
            unsafe { result::module::unload(module.cu_module) }.unwrap();
        }

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
    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [Rc<Self>], and this method increments the rc for self
    pub fn alloc_zeros<T: ValidAsZeroBits>(self: &Rc<Self>) -> Result<CudaRc<T>, CudaError> {
        let alloc = unsafe { CudaUniquePtr::alloc(self) }?;
        unsafe { result::memset_d8_async::<T>(alloc.cu_device_ptr, 0, self.cu_stream) }?;
        Ok(CudaRc {
            t_cuda: Rc::new(alloc),
            t_host: None,
        })
    }

    /// Takes ownership of `host_data`, and does an async allocation and async copy of the
    /// host data to device.
    ///
    /// # Safety
    /// 1. This takes ownership of host data, meaning any asynchronous copies from host
    /// data are safe because they are behind this struct. Since host data is an Rc,
    /// any mutations by another ref will not mutate this data.
    /// 2. The device memory is valid because the host memory is valid.
    /// 3. Self is [Rc<Self>], and this method increments the rc for self
    pub fn take<T>(self: &Rc<Self>, host_data: Rc<T>) -> Result<CudaRc<T>, CudaError> {
        let alloc = unsafe { CudaUniquePtr::alloc(self) }?;
        unsafe {
            result::memcpy_htod_async(alloc.cu_device_ptr, host_data.as_ref(), self.cu_stream)
        }?;
        Ok(CudaRc {
            t_cuda: Rc::new(alloc),
            t_host: Some(host_data),
        })
    }

    /// If host data exists, schedules a device to host copy and then synchronizes
    ///
    /// Note: This will clone the host data if there is more than 1 reference to it.
    pub(crate) fn maybe_sync_host<T: Clone>(&self, t: &mut CudaRc<T>) -> Result<(), CudaError> {
        if let Some(host_data) = &mut t.t_host {
            unsafe {
                result::memcpy_dtoh_async(
                    Rc::make_mut(host_data),
                    t.t_cuda.cu_device_ptr,
                    self.cu_stream,
                )
            }?;
            self.synchronize()?;
        }
        Ok(())
    }

    /// Synchronizes the stream.
    pub(crate) fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { result::stream::synchronize(self.cu_stream) }
    }

    /// Return the module associated with `key`.
    pub fn get_module(&self, key: &str) -> Option<&CudaModule> {
        self.modules.get(key)
    }
}

/// Wrapper around [sys::CUmodule] that also contains
/// the loaded [CudaFunction] associated with this module.
///
/// See [CudaModule::get_fn()] for retrieving function handles.
///
/// See [CudaDeviceBuilder] for how to construct these modules.
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: HashMap<&'static str, CudaFunction>,
}

impl CudaModule {
    /// Returns reference to function with `name`. If function
    /// was not already loaded into CudaModule, then `None`
    /// is returned.
    pub fn get_fn(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }
}

/// Wrapper around [sys::CUfunction] to prevent it from being cloned.
#[derive(Debug)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
}

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

/// Can launch a [CudaFunction] with the corresponding generic type `Params`.
///
/// This is impl'd multiple times for different number and types of params. In
/// general, `Params` should impl [IntoKernelParam]
///
/// # Safety
/// This is not safe really ever, because there's no garuntee that `Params`
/// will work for any [CudaFunction] passed in. Great care should be taken
/// to ensure that [CudaFunction] works with `Params` and that the correct
/// parameters have `&mut` in front of them.
///
/// **Make sure that any mutable [CudaRc] are passed with `&mut CudaRc`,
/// to ensure that the ref count is correctly maintained**
pub unsafe trait LaunchCudaFunction<Params> {
    /// Launches the [CudaFunction] with the corresponding `Params`.
    ///
    /// **Make sure that any mutable [CudaRc] are passed with `&mut CudaRc`,
    /// to ensure that the ref count is correctly maintained**
    ///
    /// # Safety
    /// This method is **very** unsafe.
    /// 1. `params` can be changed regardless of `&` or `&mut` usage.
    /// 2. `params` will be changed at some later point even after the
    /// function returns due to async
    /// 3. There are no guaruntees that the `params`
    /// are the correct number/types/order for `func`.
    unsafe fn launch_cuda_function(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), CudaError>;
}

unsafe impl<T> IntoKernelParam for &mut CudaRc<T> {
    fn into_kernel_param(self) -> *mut std::ffi::c_void {
        let ptr = Rc::make_mut(&mut self.t_cuda);
        (&mut ptr.cu_device_ptr) as *mut sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<T> IntoKernelParam for &CudaRc<T> {
    fn into_kernel_param(self) -> *mut std::ffi::c_void {
        (&self.t_cuda.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

macro_rules! impl_into_kernel_param {
    ($T:ty) => {
        unsafe impl IntoKernelParam for &$T {
            fn into_kernel_param(self) -> *mut std::ffi::c_void {
                self as *const $T as *mut std::ffi::c_void
            }
        }

        unsafe impl IntoKernelParam for &mut $T {
            fn into_kernel_param(self) -> *mut std::ffi::c_void {
                self as *mut $T as *mut std::ffi::c_void
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
unsafe impl<$($Vars: IntoKernelParam),*> LaunchCudaFunction<($($Vars, )*)> for CudaDevice {
    unsafe fn launch_cuda_function(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), CudaError> {
        let params = &mut [$(args.$Idx.into_kernel_param(), )*];
        result::launch_kernel(
            func.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.cu_stream,
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

/// A builder for [CudaDevice].
///
/// Call [CudaDeviceBuilder::new()] to start, and [CudaDeviceBuilder::build]
/// to finish.
///
/// Provides a way to specify what modules & functions to load into
/// the device via [CudaDeviceBuilder::with_precompiled_ptx()]
/// and [CudaDeviceBuilder::with_nvrtc_ptx()].
#[derive(Debug)]
pub struct CudaDeviceBuilder {
    pub(crate) ordinal: usize,
    pub(crate) precompiled_modules: Vec<PrecompiledPtxConfig>,
    pub(crate) nvrtc_modules: Vec<NvrtcConfig>,
}

#[derive(Debug)]
pub(crate) struct PrecompiledPtxConfig {
    pub(crate) key: &'static str,
    pub(crate) fname: &'static str,
    pub(crate) fn_names: Vec<&'static str>,
}

#[derive(Debug)]
pub(crate) struct NvrtcConfig {
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
            precompiled_modules: Vec::new(),
            nvrtc_modules: Vec::new(),
        }
    }

    /// Adds a path to a precompiled `.ptx` file to be loaded as a module on the device.
    ///
    /// - `key` is a unique identifier used to access the module later on with [CudaDevice::get_module()]
    /// - `path` is a file
    /// - `fn_names` is a slice of function names to load into the module during build.
    pub fn with_precompiled_ptx(
        mut self,
        key: &'static str,
        path: &'static str,
        fn_names: &[&'static str],
    ) -> Self {
        self.precompiled_modules.push(PrecompiledPtxConfig {
            key,
            fname: path,
            fn_names: fn_names.to_vec(),
        });
        self
    }

    /// Add a [Ptx] compiled with nvrtc to be loaded as a module on the device.
    ///
    /// - `key` is a unique identifier used to access the module later on with [CudaDevice::get_module()]
    /// - `ptx` contains the compilex ptx
    /// - `fn_names` is a slice of function names to load into the module during build.
    pub fn with_nvrtc_ptx(
        mut self,
        key: &'static str,
        ptx: Ptx,
        fn_names: &[&'static str],
    ) -> Self {
        self.nvrtc_modules.push(NvrtcConfig {
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
    pub fn build(mut self) -> Result<Rc<CudaDevice>, BuildError> {
        result::init().map_err(BuildError::InitError)?;

        let cu_device =
            result::device::get(self.ordinal as i32).map_err(BuildError::DeviceError)?;

        // primary context initialization
        let cu_primary_ctx =
            unsafe { result::primary_ctx::retain(cu_device) }.map_err(BuildError::ContextError)?;

        // TODO is this necessary to call?
        unsafe { result::ctx::set_current(cu_primary_ctx) }.map_err(BuildError::ContextError)?;

        // stream initialization
        let cu_stream = result::stream::create(result::stream::StreamKind::NonBlocking)
            .map_err(BuildError::StreamError)?;

        let mut modules =
            HashMap::with_capacity(self.nvrtc_modules.len() + self.precompiled_modules.len());

        for cu in self.precompiled_modules.drain(..) {
            let name_c = CString::new(cu.fname).map_err(BuildError::CStringError)?;
            let cu_module =
                result::module::load(name_c).map_err(|e| BuildError::PtxLoadingError {
                    key: cu.key,
                    cuda: e,
                })?;
            let module = Self::build_module(cu.key, cu_module, &cu.fn_names)?;
            modules.insert(cu.key, module);
        }

        for ptx in self.nvrtc_modules.drain(..) {
            let image = ptx.ptx.image.as_ptr() as *const _;
            let cu_module = unsafe { result::module::load_data(image) }.map_err(|e| {
                BuildError::NvrtcLoadingError {
                    key: ptx.key,
                    cuda: e,
                }
            })?;
            let module = Self::build_module(ptx.key, cu_module, &ptx.fn_names)?;
            modules.insert(ptx.key, module);
        }

        let device = CudaDevice {
            cu_device,
            cu_primary_ctx,
            cu_stream,
            modules,
        };
        Ok(Rc::new(device))
    }

    fn build_module(
        key: &'static str,
        cu_module: sys::CUmodule,
        fn_names: &[&'static str],
    ) -> Result<CudaModule, BuildError> {
        let mut functions = HashMap::with_capacity(fn_names.len());
        for &fn_name in fn_names.iter() {
            let fn_name_c = CString::new(fn_name).map_err(BuildError::CStringError)?;
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }
                .map_err(|e| BuildError::GetFunctionError {
                    key,
                    symbol: fn_name,
                    cuda: e,
                })?;
            functions.insert(fn_name, CudaFunction { cu_function });
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
    InitError(CudaError),
    DeviceError(CudaError),
    ContextError(CudaError),
    StreamError(CudaError),
    PtxLoadingError {
        key: &'static str,
        cuda: CudaError,
    },
    NvrtcLoadingError {
        key: &'static str,
        cuda: CudaError,
    },
    GetFunctionError {
        key: &'static str,
        symbol: &'static str,
        cuda: CudaError,
    },
    CStringError(std::ffi::NulError),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

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
    use super::*;
    use std::rc::Rc;

    #[test]
    fn test_post_build_rc_count() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        assert_eq!(Rc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_alloc_rc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.alloc_zeros::<f32>().unwrap();
        assert!(t.t_host.is_none());
        assert_eq!(Rc::strong_count(&device), 2);
        assert_eq!(Rc::strong_count(&t.t_cuda), 1);
    }

    #[test]
    fn test_post_take_rc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take(Rc::new(0.0f32)).unwrap();
        assert!(t.t_host.is_some());
        assert_eq!(Rc::strong_count(&device), 2);
        assert_eq!(Rc::strong_count(&t.t_cuda), 1);
        assert_eq!(t.t_host.as_ref().map(Rc::strong_count).unwrap(), 1);
        drop(t);
        assert_eq!(Rc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_rc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take(Rc::new(0.0f32)).unwrap();
        let r = t.clone();
        assert_eq!(Rc::strong_count(&device), 3);
        assert_eq!(Rc::strong_count(&t.t_cuda), 2);
        assert_eq!(Rc::strong_count(&r.t_cuda), 2);
        assert_eq!(t.t_host.as_ref().map(Rc::strong_count).unwrap(), 2);
        assert_eq!(r.t_host.as_ref().map(Rc::strong_count).unwrap(), 2);
        drop(t);
        assert_eq!(Rc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Rc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_into_host_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take(Rc::new(0.0f32)).unwrap();
        let r = t.clone();

        // NOTE: cuda rc was decremented, but host rc was not
        let r_host = r.into_host().unwrap();
        assert_eq!(Rc::strong_count(&device), 2);
        assert_eq!(Rc::strong_count(&t.t_cuda), 1);
        assert_eq!(t.t_host.as_ref().map(Rc::strong_count).unwrap(), 2);
        assert_eq!(Rc::strong_count(&r_host), 2);

        drop(r_host);
        assert_eq!(Rc::strong_count(&device), 2);
        assert_eq!(Rc::strong_count(&t.t_cuda), 1);
        assert_eq!(t.t_host.as_ref().map(Rc::strong_count).unwrap(), 1);
    }

    #[test]
    fn test_post_alloc_memory() {
        todo!();
    }

    #[test]
    fn test_post_cudarc_drop_memory() {
        todo!();
    }

    #[test]
    fn test_post_device_drop_memory() {
        todo!();
    }

    #[test]
    fn test_mut_into_kernel_param_no_inc_rc() {
        todo!();
    }

    #[test]
    fn test_ref_into_kernel_param_inc_rc() {
        todo!();
    }

    #[test]
    fn test_build_with_ptxs() {
        todo!();
    }

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        todo!();
    }

    #[test]
    fn test_launch_with_mut_and_ref_builtin() {
        todo!();
    }
}
