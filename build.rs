use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(not(feature = "ci-check"))]
    link_cuda();
}

#[allow(unused)]
fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    for path in libs(root().expect("Cuda root not found")) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    #[cfg(feature = "driver")]
    println!("cargo:rustc-link-lib=dylib=cuda");
    #[cfg(feature = "driver")]
    println!("cargo:rustc-link-lib=static=cudart_static");
    #[cfg(feature = "nvrtc")]
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    #[cfg(feature = "curand")]
    println!("cargo:rustc-link-lib=dylib=curand");

    println!("cargo:rerun-if-env-changed=STATIC");
    let is_static = std::env::var("STATIC").unwrap_or_else(|_| "0".to_string()) == "1";

    if is_static {
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=dylib=stdc++");
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=static=cublas_static");
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=static=cublasLt_static");
    } else {
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=dylib=cublas");
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }

    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-search=native={}", std::env::var("CUDNN_LIB").expect("Failed to read environmental variable `CUDNN_LIB`. Set `CUDNN_LIB` to `path/to/cudnn/lib`."));
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=dylib=cudnn");
}

fn root() -> Option<PathBuf> {
    let env_vars = ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"];
    let roots = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];

    let env_vars = env_vars.iter().map(std::env::var).filter_map(Result::ok);
    let roots = roots.iter().cloned().map(Into::into);
    let mut candidates = env_vars.chain(roots).map(Into::<PathBuf>::into);
    candidates.find(|path| path.join("include").join("cuda.h").is_file())
}

fn libs(root: PathBuf) -> Vec<PathBuf> {
    [
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .iter()
    .map(|&p| root.join(p))
    .filter(|p| p.is_dir())
    .collect()
}
