use cudas::nvrtc::compile::{compile_ptx, CompilationError};

fn main() -> Result<(), CompilationError> {
    let _ = compile_ptx(
        "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}",
    )?;
    println!("Compilation succeeded!");
    Ok(())
}
