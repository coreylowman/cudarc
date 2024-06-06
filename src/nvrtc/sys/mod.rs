#[cfg(feature = "cuda-11070")]
mod sys_11070;
#[cfg(feature = "cuda-11070")]
pub use sys_11070::*;

#[cfg(feature = "cuda-11080")]
mod sys_11080;
#[cfg(feature = "cuda-11080")]
pub use sys_11080::*;

#[cfg(feature = "cuda-12000")]
mod sys_12000;
#[cfg(feature = "cuda-12000")]
pub use sys_12000::*;

#[cfg(feature = "cuda-12010")]
mod sys_12010;
#[cfg(feature = "cuda-12010")]
pub use sys_12010::*;

#[cfg(feature = "cuda-12020")]
mod sys_12020;
#[cfg(feature = "cuda-12020")]
pub use sys_12020::*;

#[cfg(feature = "cuda-12030")]
mod sys_12030;
#[cfg(feature = "cuda-12030")]
pub use sys_12030::*;

#[cfg(feature = "cuda-12040")]
mod sys_12040;
#[cfg(feature = "cuda-12040")]
pub use sys_12040::*;

#[cfg(feature = "cuda-12050")]
mod sys_12050;
#[cfg(feature = "cuda-12050")]
pub use sys_12050::*;

pub unsafe fn lib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        match std::env::var("NVRTC_LIB_OVERRIDE") {
            Ok(nvrtc_lib_override) => {
                match Lib::new(libloading::library_filename(&nvrtc_lib_override)) {
                    Ok(lib) => return lib,
                    Err(err) => {
                        panic!("Failed to load {nvrtc_lib_override}; error = {err:?}");
                    }
                }
            },
            Err(_) => {
                let lib_name = "nvrtc";
                let choices = crate::get_lib_name_candidates(lib_name);
                for choice in choices.iter() {
                    if let Ok(lib) = Lib::new(libloading::library_filename(choice)) {
                        return lib;
                    }
                }
                panic!(
                    "Unable to find {lib_name} lib under the names {choices:?}. Please open GitHub issue."
                );
            },
        }
    })
}
