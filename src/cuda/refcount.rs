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
    pub(crate) t_cuda: Rc<CudaPtr<T>>,
    pub(crate) t_host: Option<Rc<T>>,
}

impl<T: Clone> CudaRc<T> {
    pub fn dup(&self) -> Result<Self, CudaError> {
        self.t_cuda.device.dup(self)
    }

    pub fn maybe_reclaim_host(mut self) -> Result<Option<Rc<T>>, CudaError> {
        self.t_cuda.device.clone().sync_host(&mut self)?;
        // NOTE: CudaAlloc drop impl is called here
        Ok(self.t_host)
    }

    pub fn reclaim_host(mut self) -> Result<Rc<T>, CudaError> {
        self.t_host.get_or_insert_with(|| {
            let layout = Layout::new::<T>();
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                Box::from_raw(ptr).into()
            }
        });
        self.t_cuda.device.clone().sync_host(&mut self)?;
        // NOTE: CudaAlloc drop impl is called here
        Ok(self.t_host.unwrap())
    }
}

#[derive(Debug)]
pub(crate) struct CudaPtr<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub device: Rc<CudaDevice>,
    marker: PhantomData<*const T>,
}

impl<T> Drop for CudaPtr<T> {
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

#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: HashMap<&'static str, CudaFunction>,
}

impl CudaModule {
    pub unsafe fn get_fn(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }
}

#[derive(Debug)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
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

pub struct CudaDeviceBuilder {
    ordinal: usize,
    precompiled_modules: Vec<PrecompiledPtxConfig>,
    nvrtc_modules: Vec<NvrtcConfig>,
}

pub(crate) struct PrecompiledPtxConfig {
    pub(crate) key: &'static str,
    pub(crate) fname: &'static str,
    pub(crate) fn_names: Vec<&'static str>,
}

pub(crate) struct NvrtcConfig {
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
        fname: &'static str,
        fn_names: &[&'static str],
    ) -> Self {
        self.precompiled_modules.push(PrecompiledPtxConfig {
            key,
            fname,
            fn_names: fn_names.iter().cloned().collect(),
        });
        self
    }

    pub fn with_nvrtc_ptx(
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

    pub fn build(mut self) -> Result<Rc<CudaDevice>, CudaError> {
        result::init()?;
        let cu_device = result::device::get(self.ordinal as i32)?;

        // primary context initialization
        let cu_primary_ctx = unsafe { result::device::primary_ctx_retain(cu_device) }?;
        unsafe { result::ctx::set_current(cu_primary_ctx) }?;

        // stream initialization
        let cu_stream =
            result::stream::create(result::stream::CUstream_flags::CU_STREAM_NON_BLOCKING)?;

        let mut modules =
            HashMap::with_capacity(self.nvrtc_modules.len() + self.precompiled_modules.len());

        // load
        for cu in self.precompiled_modules.drain(..) {
            let module = Self::build_module(result::module::load(cu.fname)?, &cu.fn_names)?;
            modules.insert(cu.key, module);
        }

        for ptx in self.nvrtc_modules.drain(..) {
            let module = Self::build_module(
                unsafe { result::module::load_data(ptx.ptx.image.as_ptr() as *const _) }?,
                &ptx.fn_names,
            )?;
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
        cu_module: sys::CUmodule,
        fn_names: &[&'static str],
    ) -> Result<CudaModule, CudaError> {
        let mut functions = HashMap::with_capacity(fn_names.len());
        for &fn_name in fn_names.iter() {
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name) }?;
            functions.insert(fn_name, CudaFunction { cu_function });
        }
        Ok(CudaModule {
            cu_module,
            functions,
        })
    }
}

impl CudaDevice {
    /// unsafe because the memory is unset
    unsafe fn alloc<T>(self: &Rc<Self>) -> Result<CudaPtr<T>, CudaError> {
        let cu_device_ptr = unsafe { result::malloc_async::<T>(self.cu_stream) }?;
        Ok(CudaPtr {
            cu_device_ptr,
            device: self.clone(),
            marker: PhantomData,
        })
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

    pub fn dup<T>(self: &Rc<Self>, src: &CudaRc<T>) -> Result<CudaRc<T>, CudaError> {
        let alloc = unsafe { self.alloc() }?;
        unsafe {
            result::memcpy_dtod_async::<T>(
                alloc.cu_device_ptr,
                src.t_cuda.cu_device_ptr,
                self.cu_stream,
            )
        }?;
        Ok(CudaRc {
            t_cuda: Rc::new(alloc),
            t_host: src.t_host.clone(),
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

    pub fn has_module(&self, key: &'static str) -> bool {
        self.modules.contains_key(key)
    }

    pub fn get_module(&self, fname: &str) -> Option<&CudaModule> {
        self.modules.get(fname)
    }
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
    fn into_kernel_param(&mut self) -> *mut std::ffi::c_void;
}

pub trait LaunchCudaFunction<Params> {
    unsafe fn launch_cuda_function(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), CudaError>;
}

unsafe impl<T> IntoKernelParam for &CudaRc<T> {
    fn into_kernel_param(&mut self) -> *mut std::ffi::c_void {
        (&self.t_cuda.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

macro_rules! impl_into_kernel_param {
    ($T:ty) => {
        unsafe impl IntoKernelParam for $T {
            fn into_kernel_param(&mut self) -> *mut std::ffi::c_void {
                self as *mut _ as *mut std::ffi::c_void
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
        mut args: ($($Vars, )*)
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
