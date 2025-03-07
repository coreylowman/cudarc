use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // You can load a function from a pre-compiled PTX like so:
    let module = ctx.load_ptx(Ptx::from_file("./examples/sin.ptx"), &["sin_kernel"])?;

    // and then retrieve the function with `get_func`
    let f = module.get_func("sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = stream.memcpy_stod(&a_host)?;
    let mut b_dev = a_dev.clone();

    // we use a buidler pattern to launch kernels.
    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&mut b_dev);
    launch_args.arg(&a_dev);
    launch_args.arg(n as i32);
    unsafe { launch_args.launch(cfg) }?;

    let a_host_2 = stream.memcpy_dtov(&a_dev)?;
    let b_host = stream.memcpy_dtov(&b_dev)?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
