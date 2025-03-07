use cudarc::driver::{CudaContext, CudaSlice, DriverError};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // unsafe initialization of unset memory
    let _: CudaSlice<f32> = unsafe { stream.alloc::<f32>(10) }?;

    // this will have memory initialized as 0
    let _: CudaSlice<f64> = stream.alloc_zeros::<f64>(10)?;

    // initialize with slices!
    let _: CudaSlice<usize> = stream.memcpy_stod(&vec![0; 10])?;
    let _: CudaSlice<u32> = stream.memcpy_stod(&[1, 2, 3])?;

    Ok(())
}
