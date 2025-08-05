#[cfg(feature = "cuda-13000")]
compile_error!("cusolver doesn't support cuda 13.0 yet, please open a PR once support has been added by cusolver team");

#[allow(warnings)]
pub mod sys;

#[cfg(test)]
mod sys_test;

pub mod result;

pub mod safe;
