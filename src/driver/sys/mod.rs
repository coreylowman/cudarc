#[cfg(feature = "cuda_11080")]
mod sys_11080;
#[cfg(feature = "cuda_11080")]
pub use sys_11080::*;

#[cfg(feature = "cuda_12000")]
mod sys_12000;
#[cfg(feature = "cuda_12000")]
pub use sys_12000::*;

#[cfg(feature = "cuda_12010")]
mod sys_12010;
#[cfg(feature = "cuda_12010")]
pub use sys_12010::*;

#[cfg(feature = "cuda_12020")]
mod sys_12020;
#[cfg(feature = "cuda_12020")]
pub use sys_12020::*;

pub unsafe fn lib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| Lib::new(libloading::library_filename("cuda")).unwrap())
}
