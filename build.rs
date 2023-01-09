fn main() {
    find_cuda_helper::include_cuda();
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=curand");
    println!("cargo:rustc-link-lib=dylib=cudart");
    // TODO maybe clean up
    println!(r"cargo:rustc-link-search=native=C:\Program Files\NVIDIA\CUDNN\v8.6\lib\x64");
    println!("cargo:rustc-link-lib=dylib=cudnn64_8");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
