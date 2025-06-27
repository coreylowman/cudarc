use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_file("./examples/sin.ptx"))?;
    let f = module.load_function("sin_kernel")?;

    let n = 3i32;
    let a_host = [1.0, 2.0, 3.0];
    let a_dev = stream.memcpy_stod(&a_host)?;
    let mut b_dev = stream.alloc_zeros::<f32>(n as usize)?;

    // we can safely create a second stream using [CudaStream::fork()].
    // This synchronizes with the source stream, so
    // the `memcpy_vtod` & `alloc_zeros` above will complete **before**
    // work on this stream can start.
    let stream2 = stream.fork()?;

    // now we launch this work on the other stream
    let mut builder = stream2.launch_builder(&f);
    builder.arg(&mut b_dev); // NOTE: tells cudarc that we are mutating this.
    builder.arg(&a_dev); // NOTE: tells cudarc that we are reading from this slice
    builder.arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;

    // cudarc automatically manages multi stream synchronization,
    // so even though we launched the above on a separate stream,
    // doing this device to host transfer will still properly synchronize.
    // a_dev doesn't need to synchronize at all since we specified it is just
    // being read from.
    // b_dev DOES need to be synchronized, because it was mutated on a different stream.
    let a_host_2 = stream.memcpy_dtov(&a_dev)?;
    let b_host = stream.memcpy_dtov(&b_dev)?;

    println!("Found {b_host:?}");
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
