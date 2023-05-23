use std::{path::PathBuf, array};

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::{Ptx, PtxCrate, compile_crate_to_ptx},
};

use std::time::Instant;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // use compile_crate_to_ptx to build kernels in pure Rust
    // uses experimental ABI_PTX
    let kernel_path: PathBuf = "examples/rust-kernel/src/lib.rs".into();
    let kernels: Vec<Ptx> = compile_crate_to_ptx(&kernel_path).unwrap();
    let kernel = kernels.first().unwrap();

    // load the ptx file...
    dev.load_ptx(kernel.clone(), "rust_kernel", &["square_kernel"])?;
    
    // and then retrieve the function with `get_func`
    let f = dev.get_func("rust_kernel", "square_kernel").unwrap();

    const GRID: u32 = 32;
    const BLOCK: u32 = 512;
    const THREADS: u32 = GRID * BLOCK;

    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    let cfg = LaunchConfig {
        grid_dim: (GRID, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // or use let cfg = LaunchConfig::for_num_elems(n);

    let a_host: [f32; THREADS as usize] = array::from_fn(|i| {
        i as f32
    });

    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    println!("squaring 0.0, ..., {}.0 with cuda kernels...", THREADS - 1);
    
    let start = Instant::now();
    unsafe { f.launch(cfg, (&a_dev, &mut b_dev, THREADS as i32)) }?;
    let duration = start.elapsed();
    println!("launch duration: {:?}", duration);
    
    let start = Instant::now();
    let b_host = dev.sync_reclaim(b_dev.clone())?;
    let duration = start.elapsed();
    println!("duration to get results: {:?}", duration);
    
    
    let a_host_2 = dev.sync_reclaim(a_dev.clone())?;
    let results: Vec<_> = a_host_2.into_iter().zip(b_host.into_iter()).collect();
    
    println!("checking some results:");
    let some_indices: [u32; 6] = [0, THREADS/10, THREADS/5, THREADS/4, THREADS/2, THREADS - 1];
    for i in some_indices {
        let (x, x_square) = results.get(i as usize).unwrap();
        println!("\t{x}^2\t = {x_square}")
    }

    // we can also manage and clean up the build ptx files with a PtxCrate
    let mut rust_ptx: PtxCrate = kernel_path.try_into().unwrap();
    rust_ptx.build_ptx().unwrap();
    let _kernel: &Ptx = rust_ptx.peek_kernels().unwrap().first().unwrap();
    println!("cleaned successfully? {:?}", rust_ptx.clean());

    Ok(())
}
