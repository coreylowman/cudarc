use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    let (major, minor): (usize, usize) = if cfg!(feature = "cuda-12050") {
        (12, 5)
    } else if cfg!(feature = "cuda-12040") {
        (12, 4)
    } else if cfg!(feature = "cuda-12030") {
        (12, 3)
    } else if cfg!(feature = "cuda-12020") {
        (12, 2)
    } else if cfg!(feature = "cuda-12010") {
        (12, 1)
    } else if cfg!(feature = "cuda-12000") {
        (12, 0)
    } else if cfg!(feature = "cuda-11080") {
        (11, 8)
    } else if cfg!(feature = "cuda-11070") {
        (11, 7)
    } else if cfg!(feature = "cuda-11060") {
        (11, 6)
    } else if cfg!(feature = "cuda-11050") {
        (11, 5)
    } else {
        #[cfg(not(feature = "cuda-version-from-build-system"))]
        panic!("Must specify one of the following features: [cuda-version-from-build-system, cuda-12050, cuda-12040, cuda-12030, cuda-12020, cuda-12010, cuda-12000, cuda-11080, cuda-11070, cuda-11060, cuda-11050]");

        #[cfg(feature = "cuda-version-from-build-system")]
        {
            let (major, minor) = cuda_version_from_build_system();
            println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");
            (major, minor)
        }
    };

    println!("cargo:rustc-env=CUDA_MAJOR_VERSION={major}");
    println!("cargo:rustc-env=CUDA_MINOR_VERSION={minor}");

    #[cfg(feature = "dynamic-linking")]
    dynamic_linking();
}

#[allow(unused)]
fn cuda_version_from_build_system() -> (usize, usize) {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to execute `nvcc`");

    if !output.status.success() {
        panic!(
            "`nvcc --version` failed.\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().nth(3).unwrap();
    let release_section = version_line.split(", ").nth(1).unwrap();
    let version_number = release_section.split(' ').nth(1).unwrap();

    match version_number {
        "12.5" => (12, 5),
        "12.4" => (12, 4),
        "12.3" => (12, 3),
        "12.2" => (12, 2),
        "12.1" => (12, 1),
        "12.0" => (12, 0),
        "11.8" => (11, 8),
        "11.7" => (11, 7),
        "11.6" => (11, 6),
        "11.5" => (11, 5),
        v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
    }
}

#[allow(unused)]
fn dynamic_linking() {
    let candidates: Vec<PathBuf> = root_candidates().collect();

    let toolkit_root = root_candidates()
        .find(|path| path.join("include").join("cuda.h").is_file())
        .unwrap_or_else(|| {
            panic!(
                "Unable to find `include/cuda.h` under any of: {:?}. Set the `CUDA_ROOT` environment variable to `$CUDA_ROOT/include/cuda.h` to override path.",
                candidates
            )
        });

    for path in lib_candidates(&toolkit_root) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    #[cfg(feature = "cudnn")]
    {
        let cudnn_root = root_candidates()
            .find(|path| path.join("include").join("cudnn.h").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "Unable to find `include/cudnn.h` under any of: {:?}. Set the `CUDNN_LIB` environment variable to `$CUDNN_LIB/include/cudnn.h` to override path.",
                    candidates
                )
            });

        for path in lib_candidates(&cudnn_root) {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    #[cfg(feature = "driver")]
    println!("cargo:rustc-link-lib=dylib=cuda");
    #[cfg(feature = "nccl")]
    println!("cargo:rustc-link-lib=dylib=nccl");
    #[cfg(feature = "nvrtc")]
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    #[cfg(feature = "curand")]
    println!("cargo:rustc-link-lib=dylib=curand");
    #[cfg(feature = "cublas")]
    println!("cargo:rustc-link-lib=dylib=cublas");
    #[cfg(any(feature = "cublas", feature = "cublaslt"))]
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=dylib=cudnn");
}

#[allow(unused)]
fn root_candidates() -> impl Iterator<Item = PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::into);
    env_vars.chain(roots).map(Into::<PathBuf>::into)
}

#[allow(unused)]
fn lib_candidates(root: &Path) -> Vec<PathBuf> {
    [
        "lib",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
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
