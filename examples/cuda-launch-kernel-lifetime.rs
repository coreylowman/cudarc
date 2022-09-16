use cudas::cuda::borrow::prelude::*;

fn main() -> Result<(), CudaError> {
    let mut dev = CudaDevice::new(0)?;

    if !dev.has_module("sin") {
        let module = dev.load_module_from_ptx_file("sin", "./examples/sin.ptx")?;
        module.load_fn("sin_kernel")?;
    }

    let module = dev.get_module("sin").unwrap();
    let f = module.get_fn("sin_kernel").unwrap();

    let a_host: Box<[f32; 3]> = Box::new([1.0, 2.0, 3.0]);

    let mut a_dev = dev.take(a_host.clone())?;
    let mut b_dev = unsafe { dev.alloc::<[f32; 3]>() }?;

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { dev.launch_cuda_function(f, cfg, (&mut b_dev, &mut a_dev, n)) }?;

    let b_host = dev.release(b_dev)?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));

    Ok(())
}
