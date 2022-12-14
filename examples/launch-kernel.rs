use cudarc::device::{CudaDeviceBuilder, LaunchConfig, LaunchCudaFunction};

fn main() {
    let dev = CudaDeviceBuilder::new(0)
        .with_ptx_from_file("sin_module", "./examples/sin.ptx", &["sin_kernel"])
        .build()
        .unwrap();

    // "sin_module" is the key used with CudaDeviceBuilder
    let module = dev.get_module("sin_module").unwrap();

    // "sin_kernel" is the name of the actual function inside the .ptx file
    let f = module.get_fn("sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = dev.sync_copy(&a_host).unwrap();
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { dev.launch_async(f, cfg, (&mut b_dev, &a_dev, &n)) }.unwrap();

    let a_host_2 = dev.sync_release(a_dev).unwrap();
    let b_host = dev.sync_release(b_dev).unwrap();

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());
}
