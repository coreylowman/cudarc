use cudarc::driver::{CudaContext, CudaSlice, DriverError};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a: CudaSlice<f64> = stream.alloc_zeros::<f64>(10)?;
    let mut b = stream.alloc_zeros::<f64>(10)?;

    // you can do device to device copies of course
    stream.memcpy_dtod(&a, &mut b)?;

    // but also host to device copys with already allocated buffers
    stream.memcpy_htod(&vec![2.0; 10], &mut b)?;
    // you can use any type of slice
    stream.memcpy_htod(&[3.0; 10], &mut b)?;

    // you can transfer back using memcpy_dtov
    let mut a_host: Vec<f64> = stream.memcpy_dtov(&a)?;
    assert_eq!(a_host, [0.0; 10]);

    let b_host = stream.memcpy_dtov(&b)?;
    assert_eq!(b_host, [3.0; 10]);

    // or transfer into a pre allocated slice
    stream.memcpy_dtoh(&b, &mut a_host)?;
    assert_eq!(a_host, b_host);

    Ok(())
}
