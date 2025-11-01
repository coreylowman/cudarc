use cudarc::{
    driver::{CudaContext, DriverError},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;

    println!("Device: {}", ctx.name()?);
    println!();

    // Load the module with the sin_kernel
    let module = ctx.load_module(Ptx::from_file("./examples/sin.ptx"))?;
    let sin_kernel = module.load_function("sin_kernel")?;

    // Query function attributes
    println!("=== Function Attributes for 'sin_kernel' ===");
    println!();

    println!("Resource Usage:");
    println!("  Registers per thread:     {}", sin_kernel.num_regs()?);
    println!(
        "  Static shared memory:     {} bytes",
        sin_kernel.shared_size_bytes()?
    );
    println!(
        "  Constant memory:          {} bytes",
        sin_kernel.const_size_bytes()?
    );
    println!(
        "  Local memory per thread:  {} bytes",
        sin_kernel.local_size_bytes()?
    );
    println!();

    println!("Limits:");
    println!(
        "  Max threads per block:    {}",
        sin_kernel.max_threads_per_block()?
    );
    println!();

    println!("Compilation Info:");
    let ptx_ver = sin_kernel.ptx_version()?;
    let bin_ver = sin_kernel.binary_version()?;
    println!(
        "  PTX version:              {}.{}",
        ptx_ver / 10,
        ptx_ver % 10
    );
    println!(
        "  Binary version:           {}.{}",
        bin_ver / 10,
        bin_ver % 10
    );
    println!();

    // Use occupancy API to get optimal launch configuration
    extern "C" fn no_dynamic_smem(_block_size: std::ffi::c_int) -> usize {
        0
    }
    let (min_grid_size, block_size) =
        sin_kernel.occupancy_max_potential_block_size(no_dynamic_smem, 0, 0, None)?;

    println!("=== Optimal Launch Configuration (sin_kernel) ===");
    println!("  Suggested block size:     {}", block_size);
    println!("  Min grid size:            {}", min_grid_size);
    println!("  Total threads per grid:   {}", min_grid_size * block_size);

    Ok(())
}
