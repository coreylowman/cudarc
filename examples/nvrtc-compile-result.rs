use cudas::nvrtc::result as nvrtc;
use std::ffi::CStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prog = nvrtc::create_program(
        "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}",
    )?;

    unsafe { nvrtc::compile_program(prog, &[]) }?;

    let log = unsafe { nvrtc::get_program_log(prog) }?;
    println!("Log: {:?}", unsafe { CStr::from_ptr(log.as_ptr()) });

    let ptx = unsafe { nvrtc::get_ptx(prog) }?;
    println!("{:?}", unsafe { CStr::from_ptr(ptx.as_ptr()) });

    Ok(())
}
