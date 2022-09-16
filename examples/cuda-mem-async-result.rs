use cudas::cuda::result as cuda;

fn main() -> Result<(), cuda::CudaError> {
    cuda::init()?;
    let dev = cuda::device::get(0)?;
    let ctx = unsafe { cuda::device::primary_ctx_retain(dev)? };
    unsafe { cuda::ctx::set_current(ctx)? };
    let stream = cuda::stream::create(cuda::stream::CUstream_flags::CU_STREAM_NON_BLOCKING)?;

    type T = [f32; 3];
    let a_host: T = [1.0, 2.0, 3.0];
    let a_dev = unsafe { cuda::malloc_async::<T>(stream) }?;
    unsafe { cuda::memcpy_htod_async(a_dev, &a_host, stream) }?;

    let mut b_host: T = [0.0; 3];
    unsafe { cuda::memcpy_dtoh_async(&mut b_host, a_dev, stream) }?;

    unsafe { cuda::free_async(a_dev, stream) }?;
    unsafe { cuda::stream::synchronize(stream) }?;

    println!("{:?}", b_host);

    unsafe { cuda::stream::destroy(stream) }?;
    unsafe { cuda::device::primary_ctx_release(dev) }?;

    Ok(())
}
