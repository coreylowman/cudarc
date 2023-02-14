fn main() {
    #[cfg(not(feature = "ci-check"))]
    {
        find_cuda_helper::include_cuda();
        println!("cargo:rustc-link-lib=dylib=nvrtc");
        println!("cargo:rustc-link-lib=dylib=curand");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }
}
