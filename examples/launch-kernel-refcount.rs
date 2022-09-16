use cudas::cuda::refcount::*;
use std::rc::Rc;

fn main() -> Result<(), CudaError> {
    let dev = CudaDeviceBuilder::new(0)
        .with_precompiled_ptx("sin", "./examples/sin.ptx", &["sin_kernel"])
        .build()?;

    let module = dev.get_module("sin").unwrap();
    let f = unsafe { module.get_fn("sin_kernel").unwrap() };

    let a_host: Rc<[f32; 3]> = Rc::new([1.0, 2.0, 3.0]);

    let a_dev = dev.take(a_host.clone())?;
    let b_dev = a_dev.dup()?;

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { dev.launch_cuda_function(f, cfg, (&b_dev, &a_dev, n)) }?;

    let b_host = b_dev.reclaim_host()?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));

    Ok(())
}
