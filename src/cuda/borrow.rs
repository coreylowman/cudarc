use super::result;
use super::sys;
use std::alloc::alloc_zeroed;
use std::alloc::Layout;
use std::collections::HashMap;
use std::marker::PhantomData;

pub mod prelude {
    pub use super::result::CudaError;
    pub use super::*;
}

#[derive(Debug)]
pub struct CudaDevice {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_primary_ctx: sys::CUcontext,
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) loaded_modules: HashMap<&'static str, CudaModule>,
}

#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: HashMap<&'static str, CudaFunction>,
}

#[derive(Debug)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
}

#[derive(Debug)]
pub struct LinkedAlloc<'device, T> {
    pub(crate) gpu_data: CudaAlloc<'device, T>,
    pub(crate) cpu_data: Option<Box<T>>,
}

impl<'a, T: Clone> LinkedAlloc<'a, T> {
    pub fn dup(&self) -> Result<Self, result::CudaError> {
        self.gpu_data.device.dup(self)
    }

    pub fn maybe_reclaim_host(mut self) -> Result<Option<Box<T>>, result::CudaError> {
        self.gpu_data.device.maybe_sync_host(&mut self)?;
        // NOTE: CudaAlloc drop impl is called here
        Ok(self.cpu_data)
    }

    pub fn reclaim_host(mut self) -> Result<Box<T>, result::CudaError> {
        self.cpu_data.get_or_insert_with(|| {
            let layout = Layout::new::<T>();
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                Box::from_raw(ptr)
            }
        });
        self.gpu_data.device.maybe_sync_host(&mut self)?;
        // NOTE: CudaAlloc drop impl is called here
        Ok(self.cpu_data.unwrap())
    }
}

#[derive(Debug)]
pub struct CudaAlloc<'device, T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) device: &'device CudaDevice,
    marker: PhantomData<*const T>,
}

impl<'a, T> Drop for CudaAlloc<'a, T> {
    fn drop(&mut self) {
        unsafe { result::free_async(self.cu_device_ptr, self.device.cu_stream) }.unwrap();
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        for (_, module) in self.loaded_modules.drain() {
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
    pub fn new(ordinal: usize) -> Result<Self, result::CudaError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        let cu_primary_ctx = unsafe { result::device::primary_ctx_retain(cu_device) }?;
        unsafe { result::ctx::set_current(cu_primary_ctx) }?;
        let cu_stream =
            result::stream::create(result::stream::CUstream_flags::CU_STREAM_NON_BLOCKING)?;
        Ok(Self {
            cu_device,
            cu_primary_ctx,
            cu_stream,
            loaded_modules: HashMap::new(),
        })
    }

    /// unsafe becuase it the memory is unset
    unsafe fn alloc<T>(&self) -> Result<CudaAlloc<T>, result::CudaError> {
        let cu_device_ptr = unsafe { result::malloc_async::<T>(self.cu_stream) }?;
        Ok(CudaAlloc {
            cu_device_ptr,
            device: self,
            marker: PhantomData,
        })
    }

    /// Unsafe because it memsets all allocated memory to 0, and T may not be valid.
    pub unsafe fn alloc_zeros<T>(&self) -> Result<LinkedAlloc<T>, result::CudaError> {
        let alloc = self.alloc()?;
        unsafe { result::memset_d8_async::<T>(alloc.cu_device_ptr, 0, self.cu_stream) }?;
        Ok(LinkedAlloc {
            gpu_data: alloc,
            cpu_data: None,
        })
    }

    pub fn take<T>(&self, host_data: Box<T>) -> Result<LinkedAlloc<T>, result::CudaError> {
        let alloc = unsafe { self.alloc()? };
        unsafe {
            result::memcpy_htod_async(alloc.cu_device_ptr, host_data.as_ref(), self.cu_stream)
        }?;
        Ok(LinkedAlloc {
            gpu_data: alloc,
            cpu_data: Some(host_data),
        })
    }

    pub fn dup<T: Clone>(&self, src: &LinkedAlloc<T>) -> Result<LinkedAlloc<T>, result::CudaError> {
        let alloc = unsafe { self.alloc() }?;
        unsafe {
            result::memcpy_dtod_async::<T>(
                alloc.cu_device_ptr,
                src.gpu_data.cu_device_ptr,
                self.cu_stream,
            )
        }?;
        Ok(LinkedAlloc {
            gpu_data: alloc,
            cpu_data: src.cpu_data.clone(),
        })
    }

    pub fn maybe_sync_host<T>(&self, t: &mut LinkedAlloc<T>) -> Result<(), result::CudaError> {
        if let Some(host_data) = &mut t.cpu_data {
            unsafe {
                result::memcpy_dtoh_async(
                    host_data.as_mut(),
                    t.gpu_data.cu_device_ptr,
                    self.cu_stream,
                )
            }?;
            self.synchronize()?;
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), result::CudaError> {
        unsafe { result::stream::synchronize(self.cu_stream) }
    }

    pub fn has_module(&self, key: &'static str) -> bool {
        self.loaded_modules.contains_key(key)
    }

    pub fn load_module_from_ptx_file(
        &mut self,
        key: &'static str,
        fname: &'static str,
    ) -> Result<&mut CudaModule, result::CudaError> {
        let cu_module = result::module::load(fname)?;
        self.insert_module(key, cu_module);
        Ok(self.loaded_modules.get_mut(key).unwrap())
    }

    pub unsafe fn load_module_from_ptx_nvrtc(
        &mut self,
        key: &'static str,
        image: *const std::ffi::c_char,
    ) -> Result<&mut CudaModule, result::CudaError> {
        let cu_module = result::module::load_data(image as *const _)?;
        self.insert_module(key, cu_module);
        Ok(self.loaded_modules.get_mut(key).unwrap())
    }

    fn insert_module(&mut self, key: &'static str, cu_module: sys::CUmodule) {
        self.loaded_modules.insert(
            key,
            CudaModule {
                cu_module,
                functions: HashMap::with_capacity(1),
            },
        );
    }

    pub fn get_module(&self, fname: &str) -> Option<&CudaModule> {
        self.loaded_modules.get(fname)
    }
}

impl CudaModule {
    pub fn load_fn(&mut self, name: &'static str) -> Result<(), result::CudaError> {
        let cu_function = unsafe { result::module::get_function(self.cu_module, name) }?;
        self.functions.insert(name, CudaFunction { cu_function });
        Ok(())
    }

    pub fn get_fn(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
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

pub trait LaunchCudaFunction<Args> {
    unsafe fn launch_cuda_function(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        args: Args,
    ) -> Result<(), result::CudaError>;
}

pub trait IntoKernelParam {
    fn into_kernel_param(&mut self) -> *mut std::ffi::c_void;
}

impl<'a, T> IntoKernelParam for &mut LinkedAlloc<'a, T> {
    fn into_kernel_param(&mut self) -> *mut std::ffi::c_void {
        (&mut self.gpu_data.cu_device_ptr) as *mut sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

impl<'a, T> IntoKernelParam for &LinkedAlloc<'a, T> {
    fn into_kernel_param(&mut self) -> *mut std::ffi::c_void {
        (&self.gpu_data.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

macro_rules! impl_into_kernel_param {
    ($T:ty) => {
        impl IntoKernelParam for $T {
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
    ) -> Result<(), result::CudaError> {
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
