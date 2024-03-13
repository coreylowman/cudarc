#[cfg(feature = "cuda_version_11_8")]
mod sys_11080;
#[cfg(feature = "cuda_version_11_8")]
pub use sys_11080::*;

#[cfg(feature = "cuda_version_12_2")]
mod sys_12020;
#[cfg(feature = "cuda_version_12_2")]
pub use sys_12020::*;
