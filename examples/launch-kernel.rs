use cudarc::driver::{CudaDeviceBuilder, DriverError, LaunchAsync, LaunchConfig};

fn main() -> Result<(), DriverError> {
    let dev = CudaDeviceBuilder::new(0)
        .with_ptx_from_file("./examples/sin.ptx", "sin_module", &["sin_kernel"])
        .build()
        .unwrap();

    let f = dev.get_func("sin_module", "sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }?;

    let a_host_2 = dev.reclaim_sync(a_dev)?;
    let b_host = dev.reclaim_sync(b_dev)?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
