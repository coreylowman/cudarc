use cudas::cuda::refcount::*;
use std::rc::Rc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = CudaDeviceBuilder::new(0)
        .with_precompiled_ptx("sin_module", "./examples/sin.ptx", &["sin_kernel"])
        .build()?;

    // "sin_module" is the key used with CudaDeviceBuilder
    let module = dev.get_module("sin_module").unwrap();

    // "sin_kernel" is the name of the actual function inside the .ptx file
    let f = module.get_fn("sin_kernel").unwrap();

    let a_host: Rc<[f32; 3]> = Rc::new([1.0, 2.0, 3.0]);

    let a_dev = dev.take(a_host.clone())?;
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { dev.launch_cuda_function(f, cfg, (&mut b_dev, &a_dev, &n)) }?;

    let a_host_2 = a_dev.into_host()?;
    let b_host = b_dev.into_host()?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(a_host.as_ref(), a_host_2.as_ref());

    Ok(())
}
