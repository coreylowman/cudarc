#[cfg(feature = "cuda-11040")]
mod sys_11040;
#[cfg(feature = "cuda-11040")]
pub use sys_11040::*;
#[cfg(feature = "cuda-11050")]
mod sys_11050;
#[cfg(feature = "cuda-11050")]
pub use sys_11050::*;
#[cfg(feature = "cuda-11060")]
mod sys_11060;
#[cfg(feature = "cuda-11060")]
pub use sys_11060::*;
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
#[cfg(feature = "cuda-12060")]
mod sys_12060;
#[cfg(feature = "cuda-12060")]
pub use sys_12060::*;
#[cfg(feature = "cuda-12080")]
mod sys_12080;
#[cfg(feature = "cuda-12080")]
pub use sys_12080::*;

pub unsafe fn culib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let lib_names = ["cublas"];
        let choices: Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = Lib::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}

mod adapter;
pub use adapter::*;
