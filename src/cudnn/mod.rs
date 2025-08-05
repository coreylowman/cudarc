#[cfg(feature = "cuda-13000")]
compile_error!("cudnn doesn't support cuda 13.0 yet, please open a PR once support has been added by cudnn team");

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
