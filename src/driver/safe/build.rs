use crate::driver::{result, sys};
use crate::nvrtc::Ptx;

use super::core::{CudaDevice, CudaModule};

use alloc::ffi::{CString, NulError};
use spin::RwLock;
use std::{collections::BTreeMap, sync::Arc, vec::Vec};

impl CudaDevice {
    /// Dynamically load a set of [crate::driver::CudaFunction] from a ptx file.
    /// See [crate::driver::CudaDeviceBuilder::with_ptx_from_file].
    pub fn load_ptx_from_file(
        self: &Arc<Self>,
        ptx_path: &'static str,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), BuildError> {
        let m = CudaDeviceBuilder::build_module_from_ptx_file(ptx_path, module_name, func_names)?;
        {
            let mut modules = self.modules.write();
            modules.insert(module_name, m);
        }
        Ok(())
    }

    /// Dynamically load a set of [crate::driver::CudaFunction] from a jit compiled ptx.
    /// See [crate::driver::CudaDeviceBuilder::with_ptx]
    pub fn load_ptx(
        self: &Arc<Self>,
        ptx: Ptx,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), BuildError> {
        let m = CudaDeviceBuilder::build_module_from_ptx(ptx, module_name, func_names)?;
        {
            let mut modules = self.modules.write();
            modules.insert(module_name, m);
        }
        Ok(())
    }
}

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

        let free_stream = result::stream::create(result::stream::StreamKind::NonBlocking)
            .map_err(BuildError::StreamError)?;

        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .map_err(BuildError::DeviceError)?;

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
            stream: std::ptr::null_mut(),
            free_stream,
            event,
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
        let cu_module = match ptx {
            Ptx::Image(image) => unsafe { result::module::load_data(image.as_ptr() as *const _) },
            Ptx::Src(src) => {
                let c_src = CString::new(src).unwrap();
                unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
            }
        }
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
    InitError(result::DriverError),
    DeviceError(result::DriverError),
    ContextError(result::DriverError),
    StreamError(result::DriverError),
    PtxLoadingError {
        key: &'static str,
        cuda: result::DriverError,
    },
    NvrtcLoadingError {
        key: &'static str,
        cuda: result::DriverError,
    },
    GetFunctionError {
        key: &'static str,
        symbol: &'static str,
        cuda: result::DriverError,
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::driver::DeviceSlice;

    #[test]
    fn test_device_copy_to_views() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();

        let smalls = [
            dev.htod_copy(std::vec![-1.0f32, -0.8]).unwrap(),
            dev.htod_copy(std::vec![-0.6, -0.4]).unwrap(),
            dev.htod_copy(std::vec![-0.2, 0.0]).unwrap(),
            dev.htod_copy(std::vec![0.2, 0.4]).unwrap(),
            dev.htod_copy(std::vec![0.6, 0.8]).unwrap(),
        ];
        let mut big = dev.alloc_zeros::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.try_slice_mut(offset..offset + small.len()).unwrap();
            dev.dtod_copy(small, &mut sub).unwrap();
            offset += small.len();
        }

        assert_eq!(
            dev.reclaim_sync(big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }
}
