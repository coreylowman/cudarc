//! [Cufile] wraps around [cuFILE](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) via:
//! 
//! 1. Instantiate a new handle to the api with [Cufile::new()]
//! 2. Register a file with [Cufile::register()], this accepts a [std::fs::File].
//! 3. Read/write from filesystem using [FileHandle::sync_read], [FileHandle::sync_write], [crate::driver::CudaStream::memcpy_dtof()], [crate::driver::CudaStream::memcpy_ftod()].
//! 
//! Note that all safe apis work with [crate::driver::DevicePtr] and [crate::driver::DevicePtrMut], meaning they accept both
//! [crate::driver::CudaSlice] and [crate::driver::CudaView]/[crate::driver::CudaViewMut].

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
