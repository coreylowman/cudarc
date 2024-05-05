
use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

const KERNEL_SRC: &str = "
__device__ float global_array[2];
__constant__ int const_array[2];
extern \"C\" __global__ void kernel(int i) {
    printf(\"global_array value  %f %f\\n\", global_array[0], global_array[1]);
    printf(\"const_array value  %d %d\\n\", const_array[0], const_array[1]);

    global_array[0] = 1.1;
    global_array[1] = 2.2;
}
";

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    let ptx = compile_ptx_with_opts(
        KERNEL_SRC,
        CompileOptions {
            arch: Some("sm_50"),
            ..Default::default()
        },
    )
    .unwrap();

    dev.load_ptx(ptx, "module", &["kernel"])?;

    let f = dev.get_func("module", "kernel").unwrap();

    // get symbol pointer
    let (mut d_global,_)= dev.get_symbol_ptr::<f32>("module", "global_array")?;

    // copy host data to device by symbol pointer (__device__)
    dev.htod_symbol_copy_into::<f32>(vec![3.14, 0.44],&mut d_global)?;

    // copy host data to device by module name and symobl name (__constant__)
    dev.htod_symbol_copy::<i32>("module", "const_array" ,vec![11, 22])?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let _ = f.launch(cfg, (1,));
    }

    dev.synchronize()?;

    // copy device symbol data to host
    let mut h_global = [0f32;2];
    let _ = dev.dtod_sync_symbol_copy_into(&d_global, &mut h_global);
    assert_eq!(h_global, [1.1, 2.2]);

    // copy device symbol data to host
    let h_const = dev.dtod_sync_symbol_copy::<i32>("module", "const_array")?;
    assert_eq!(h_const, vec![11, 22]);

    Ok(())

}
