use super::{result, sys};

pub use result::DriverError;

pub fn start() -> Result<(), DriverError> {
    unsafe { sys::cuProfilerStart() }.result()
}

pub fn stop() -> Result<(), DriverError> {
    unsafe { sys::cuProfilerStop() }.result()
}
