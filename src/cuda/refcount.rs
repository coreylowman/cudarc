use super::result;
use super::sys;
use crate::nvrtc::compile::Ptx;
use std::alloc::alloc_zeroed;
use std::alloc::Layout;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

pub use result::CudaError;

pub unsafe trait Zeroable {}
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for i64 {}
unsafe impl Zeroable for isize {}
unsafe impl Zeroable for u8 {}
unsafe impl Zeroable for u16 {}
unsafe impl Zeroable for u32 {}
unsafe impl Zeroable for u64 {}
unsafe impl Zeroable for usize {}
unsafe impl Zeroable for f32 {}
unsafe impl Zeroable for f64 {}
unsafe impl<T: Zeroable, const M: usize> Zeroable for [T; M] {}

#[derive(Debug, Clone)]
pub struct CudaRc<T> {
    pub(crate) t_cuda: Rc<CudaUniquePtr<T>>,
    pub(crate) t_host: Option<Rc<T>>,
}

impl<T> CudaRc<T> {
    pub fn device(&self) -> &Rc<CudaDevice> {
        &self.t_cuda.device
    }
}

impl<T: Clone> CudaRc<T> {
    pub fn maybe_into_host(mut self) -> Result<Option<Rc<T>>, CudaError> {
        self.t_cuda.device.clone().sync_host(&mut self)?;
        Ok(self.t_host)
    }

    pub fn into_host(mut self) -> Result<Rc<T>, CudaError> {
        self.t_host.get_or_insert_with(|| {
            let layout = Layout::new::<T>();
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                Box::from_raw(ptr).into()
            }
        });
        self.t_cuda.device.clone().sync_host(&mut self)?;
        Ok(self.t_host.unwrap())
    }
}

#[derive(Debug)]
pub(crate) struct CudaUniquePtr<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub device: Rc<CudaDevice>,
    marker: PhantomData<*const T>,
}

impl<T> Clone for CudaUniquePtr<T> {
    fn clone(&self) -> Self {
        self.device.dup(self).unwrap()
    }
}

impl<T> Drop for CudaUniquePtr<T> {
    fn drop(&mut self) {
        unsafe { result::free_async(self.cu_device_ptr, self.device.cu_stream) }.unwrap();
    }
}

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
            unsafe { result::device::primary_ctx_release(self.cu_device) }.unwrap();
        }
    }
}

impl CudaDevice {
    /// unsafe because the memory is unset
    unsafe fn alloc<T>(self: &Rc<Self>) -> Result<CudaUniquePtr<T>, CudaError> {
        let cu_device_ptr = unsafe { result::malloc_async::<T>(self.cu_stream) }?;
        Ok(CudaUniquePtr {
            cu_device_ptr,
            device: self.clone(),
            marker: PhantomData,
        })
    }

    fn dup<T>(self: &Rc<Self>, src: &CudaUniquePtr<T>) -> Result<CudaUniquePtr<T>, CudaError> {
        let alloc = unsafe { self.alloc() }?;
        unsafe {
            result::memcpy_dtod_async::<T>(alloc.cu_device_ptr, src.cu_device_ptr, self.cu_stream)
        }?;
        Ok(alloc)
    }

    pub fn alloc_zeros<T: Zeroable>(self: &Rc<Self>) -> Result<CudaRc<T>, CudaError> {
        let alloc = unsafe { self.alloc() }?;
        unsafe { result::memset_d8_async::<T>(alloc.cu_device_ptr, 0, self.cu_stream) }?;
        Ok(CudaRc {
            t_cuda: Rc::new(alloc),
            t_host: None,
        })
    }

    pub fn take<T>(self: &Rc<Self>, host_data: Rc<T>) -> Result<CudaRc<T>, CudaError> {
        let alloc = unsafe { self.alloc() }?;
        unsafe {
            result::memcpy_htod_async(alloc.cu_device_ptr, host_data.as_ref(), self.cu_stream)
        }?;
        Ok(CudaRc {
            t_cuda: Rc::new(alloc),
            t_host: Some(host_data),
        })
    }

    pub fn sync_host<T: Clone>(&self, t: &mut CudaRc<T>) -> Result<(), CudaError> {
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

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { result::stream::synchronize(self.cu_stream) }
    }

    pub fn get_module(&self, key: &str) -> Option<&CudaModule> {
        self.modules.get(key)
    }
}

#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: HashMap<&'static str, CudaFunction>,
}

impl CudaModule {
    pub fn get_fn(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }
}

#[derive(Debug)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
}

#[derive(Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub fn for_num_elems(n: u32) -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (n, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

pub unsafe trait IntoKernelParam {
    fn into_kernel_param(self) -> *mut std::ffi::c_void;
}

pub trait LaunchCudaFunction<Params> {
    /// # Safety
    ///
    /// This method is **very** unsafe.
    /// **`params` can be changed regardless of `&` or `&mut` usage.**
    ///
    /// Additionally, there are no guaruntees that the `params`
    /// are the correct number or types or order for `func`.
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
impl<$($Vars: IntoKernelParam),*> LaunchCudaFunction<($($Vars, )*)> for CudaDevice {
    unsafe fn launch_cuda_function(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), CudaError> {
        let params = &mut [$(args.$Idx.into_kernel_param(), )*];
        unsafe {
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
}
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);

#[derive(Debug)]
pub struct CudaDeviceBuilder {
    ordinal: usize,
    precompiled_modules: Vec<PrecompiledPtxConfig>,
    nvrtc_modules: Vec<NvrtcConfig>,
}

#[derive(Debug)]
pub struct PrecompiledPtxConfig {
    pub(crate) key: &'static str,
    pub(crate) fname: &'static str,
    pub(crate) fn_names: Vec<&'static str>,
}

#[derive(Debug)]
pub struct NvrtcConfig {
    pub(crate) key: &'static str,
    pub(crate) ptx: Ptx,
    pub(crate) fn_names: Vec<&'static str>,
}

impl CudaDeviceBuilder {
    pub fn new(ordinal: usize) -> Self {
        Self {
            ordinal,
            precompiled_modules: Vec::new(),
            nvrtc_modules: Vec::new(),
        }
    }

    pub fn with_precompiled_ptx(
        mut self,
        key: &'static str,
        path: &'static str,
        fn_names: &[&'static str],
    ) -> Self {
        self.precompiled_modules.push(PrecompiledPtxConfig {
            key,
            fname: path,
            fn_names: fn_names.iter().cloned().collect(),
        });
        self
    }

    pub fn with_nvrtc_module(
        mut self,
        key: &'static str,
        ptx: Ptx,
        fn_names: &[&'static str],
    ) -> Self {
        self.nvrtc_modules.push(NvrtcConfig {
            key,
            ptx,
            fn_names: fn_names.iter().cloned().collect(),
        });
        self
    }

    pub fn build(mut self) -> Result<Rc<CudaDevice>, BuildError> {
        result::init().map_err(BuildError::InitError)?;

        let cu_device =
            result::device::get(self.ordinal as i32).map_err(BuildError::OrdinalError)?;

        // primary context initialization
        let cu_primary_ctx = unsafe { result::device::primary_ctx_retain(cu_device) }
            .map_err(BuildError::ContextError)?;

        unsafe { result::ctx::set_current(cu_primary_ctx) }.map_err(BuildError::ContextError)?;

        // stream initialization
        let cu_stream =
            result::stream::create(result::stream::CUstream_flags::CU_STREAM_NON_BLOCKING)
                .map_err(BuildError::StreamError)?;

        let mut modules =
            HashMap::with_capacity(self.nvrtc_modules.len() + self.precompiled_modules.len());

        for cu in self.precompiled_modules.drain(..) {
            let cu_module =
                result::module::load(cu.fname).map_err(|e| BuildError::PtxLoadingError {
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
            let cu_function =
                unsafe { result::module::get_function(cu_module, fn_name) }.map_err(|e| {
                    BuildError::GetFunctionError {
                        key,
                        symbol: fn_name,
                        cuda: e,
                    }
                })?;
            functions.insert(fn_name, CudaFunction { cu_function });
        }
        Ok(CudaModule {
            cu_module,
            functions,
        })
    }
}

#[derive(Debug)]
pub enum BuildError {
    InitError(CudaError),
    OrdinalError(CudaError),
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
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BuildError {}
