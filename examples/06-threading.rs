use cudarc::driver::*;
use cudarc::nvrtc::compile_ptx;

use std::thread;

const KERNEL_SRC: &str = "
extern \"C\" __global__ void hello_world(int i) {
    printf(\"Hello from the cuda kernel in thread %d\\n\", i);
}
";

fn main() -> Result<(), DriverError> {
    {
        // Option 1: sharing ctx & module between threads
        thread::scope(|s| {
            let ptx = compile_ptx(KERNEL_SRC).unwrap();
            let ctx = CudaContext::new(0)?;
            let module = ctx.load_module(ptx)?;
            for i in 0..10i32 {
                let thread_ctx = ctx.clone();
                let thread_module = module.clone();
                s.spawn(move || {
                    let stream = thread_ctx.default_stream();
                    let f = thread_module.load_function("hello_world")?;
                    unsafe {
                        stream
                            .launch_builder(&f)
                            .arg(i)
                            .launch(LaunchConfig::for_num_elems(1))
                    }
                });
            }
            Ok(())
        })?;
    }

    {
        // Option 2: initializing different context in each
        // Note that this will still schedule to the same stream since we are using the
        // default stream here on the same device.
        thread::scope(move |s| {
            for i in 0..10i32 {
                s.spawn(move || {
                    let ptx = compile_ptx(KERNEL_SRC).unwrap();
                    let ctx = CudaContext::new(0)?;
                    let module = ctx.load_module(ptx)?;
                    let stream = ctx.default_stream();
                    let f = module.load_function("hello_world")?;
                    unsafe {
                        stream
                            .launch_builder(&f)
                            .arg(i)
                            .launch(LaunchConfig::for_num_elems(1))
                    }
                });
            }
            Ok(())
        })?;
    }

    Ok(())
}
