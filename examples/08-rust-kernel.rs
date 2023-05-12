use std::path::PathBuf;

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::PtxCrate,
};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // use compile_crate_to_ptx to build and rust kernels in pure rust
    // uses experimental ABI_PTX
    let kernel_path: PathBuf = "examples/rust-kernel/src/lib.rs".into();
    let kernels = PtxCrate::compile_crate_to_ptx(&kernel_path).unwrap();
    let kernel = kernels.first().unwrap();

    // load the ptx file...
    println!("loading...");
    dev.load_ptx(kernel.clone(), "kernel", &["kernel"])?;
    println!("loaded!");

    // and then retrieve the function with `get_func`
    let f = dev.get_func("kernel", "kernel").unwrap();

    let a_host = [1, 2, 3];

    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&mut b_dev, n as i32)) }?;

    let a_host_2 = dev.sync_reclaim(a_dev.clone())?;
    let b_host = dev.sync_reclaim(b_dev.clone())?;

    println!("a_host {a_host:?}");
    println!("b_host {b_host:?}");
    println!("a_host_2 {a_host_2:?}");
    
    // we can also manage and clean up the build ptx files with a PtxCrate
    let mut rust_ptx: PtxCrate = kernel_path.try_into().unwrap();
    rust_ptx.build_ptx().unwrap();
    let _kernel = rust_ptx.ptx_files().unwrap().first().unwrap();
    println!("cleaned successfully? {:?}", rust_ptx.clean());

    Ok(())
}
