use crate::{
    driver::result,
    nvrtc::{Ptx, PtxKind},
};

use super::core::{CudaContext, CudaFunction, CudaModule};

use std::ffi::CString;
use std::{collections::BTreeMap, sync::Arc};

impl CudaContext {
    /// Dynamically load a set of [crate::driver::CudaFunction] from a jit compiled ptx.
    ///
    /// - `ptx` contains the compiled ptx
    /// - `func_names` is a slice of function names to load into the module during build.
    pub fn load_ptx(
        self: &Arc<Self>,
        ptx: Ptx,
        func_names: &[&str],
    ) -> Result<Arc<CudaModule>, result::DriverError> {
        self.bind_to_thread()?;

        let cu_module = match ptx.0 {
            PtxKind::Image(image) => unsafe {
                result::module::load_data(image.as_ptr() as *const _)
            },
            PtxKind::Src(src) => {
                let c_src = CString::new(src).unwrap();
                unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
            }
            PtxKind::File(path) => {
                let name_c = CString::new(path.to_str().unwrap()).unwrap();
                result::module::load(name_c)
            }
        }?;
        let mut functions = BTreeMap::new();
        for &fn_name in func_names.iter() {
            let fn_name_c = CString::new(fn_name).unwrap();
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }?;
            functions.insert(fn_name.into(), cu_function);
        }
        Ok(Arc::new(CudaModule {
            cu_module,
            functions,
            ctx: self.clone(),
        }))
    }
}

impl CudaModule {
    /// Returns reference to function with `name`. If function
    /// was not already loaded into CudaModule, then `None`
    /// is returned.
    pub fn get_func(self: &Arc<Self>, name: &str) -> Option<CudaFunction> {
        self.functions
            .get(name)
            .cloned()
            .map(|cu_function| CudaFunction {
                cu_function,
                module: self.clone(),
            })
    }

    pub fn has_func(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}
