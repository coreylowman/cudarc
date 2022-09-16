use cudas::cuda::result as cuda;

fn main() -> Result<(), cuda::CudaError> {
    cuda::init()?;
    let dev = cuda::device::get(0)?;
    let ctx = unsafe { cuda::device::primary_ctx_retain(dev) }?;
    unsafe { cuda::ctx::set_current(ctx)? };
    let stream = cuda::stream::create(cuda::stream::CUstream_flags::CU_STREAM_NON_BLOCKING)?;

    let module = cuda::module::load("./examples/sin.ptx")?;
    let f = unsafe { cuda::module::get_function(module, "sin_kernel") }?;

    type T = [f32; 3];
    let a_host: T = [1.0, 2.0, 3.0];
    let mut b_host: T = [0.0; 3];

    let mut a_dev = unsafe { cuda::malloc_async::<T>(stream) }?;
    let mut b_dev = unsafe { cuda::malloc_async::<T>(stream) }?;
    unsafe { cuda::memcpy_htod_async(a_dev, &a_host, stream) }?;
    unsafe { cuda::memset_d8_async::<T>(b_dev, 0, stream) }?;

    unsafe {
        cuda::launch_kernel(
            f,
            (1, 1, 1),
            (3, 1, 1),
            0,
            stream,
            &mut [
                &mut b_dev as *mut _ as *mut std::ffi::c_void,
                &mut a_dev as *mut _ as *mut std::ffi::c_void,
                &mut 3 as *mut _ as *mut std::ffi::c_void,
            ],
        )
    }?;

    unsafe { cuda::memcpy_dtoh_async(&mut b_host, b_dev, stream) }?;

    unsafe { cuda::stream::synchronize(stream) }?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));

    unsafe { cuda::module::unload(module) }?;
    unsafe { cuda::stream::destroy(stream) }?;
    unsafe { cuda::device::primary_ctx_release(dev) }?;

    Ok(())
}
