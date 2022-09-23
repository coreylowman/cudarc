//! Wrappers around the [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
//! in two levels: [result], and [sys].
//! See [crate::rng] for safe wrappers.

pub mod result;
#[allow(warnings)]
pub mod sys;
