use std::path::PathBuf;

const TYPICAL_CUDA_PATH_ENV_VARS: [&str; 5] = [
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
    "CUDNN_LIB",
];

fn main() {
    #[cfg(all(
        not(feature = "dynamic-linking"),
        not(feature = "static-linking"),
        not(feature = "dynamic-loading")
    ))]
    panic!("None between `dynamic-loading`, `dynamic-linking` and `static-linking` features are active, this is a bug");
    #[cfg(all(feature = "dynamic-linking", feature = "static-linking"))]
    panic!("Both `dynamic-linking` and `static-linking` features are active, this is a bug");
    #[cfg(all(feature = "dynamic-loading", feature = "static-linking"))]
    panic!("Both `dynamic-loading` and `static-linking` features are active, this is a bug");
    #[cfg(all(feature = "dynamic-loading", feature = "dynamic-linking"))]
    panic!("Both `dynamic-loading` and `dynamic-linking` features are active, this is a bug");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");
    TYPICAL_CUDA_PATH_ENV_VARS
        .iter()
        .for_each(|var| println!("cargo:rerun-if-env-changed={var}"));

    let (major, minor): (usize, usize) = if let Ok(version) = std::env::var("CUDARC_CUDA_VERSION") {
        let (major, minor) = match version.as_str() {
            "13000" => (13, 0),
            "12090" => (12, 9),
            "12080" => (12, 8),
            "12060" => (12, 6),
            "12050" => (12, 5),
            "12040" => (12, 4),
            "12030" => (12, 3),
            "12020" => (12, 2),
            "12010" => (12, 1),
            "12000" => (12, 0),
            "11080" => (11, 8),
            "11070" => (11, 7),
            "11060" => (11, 6),
            "11050" => (11, 5),
            "11040" => (11, 4),
            v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
        };
        println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");
        (major, minor)
    } else if cfg!(feature = "cuda-13000") {
        (13, 0)
    } else if cfg!(feature = "cuda-12090") {
        (12, 9)
    } else if cfg!(feature = "cuda-12080") {
        (12, 8)
    } else if cfg!(feature = "cuda-12060") {
        (12, 6)
    } else if cfg!(feature = "cuda-12050") {
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
    } else if cfg!(feature = "cuda-11040") {
        (11, 4)
    } else {
        #[cfg(not(feature = "cuda-version-from-build-system"))]
        panic!("Must specify one of the following features: [cuda-version-from-build-system, cuda-13000, cuda-12090, cuda-12080, cuda-12060, cuda-12050, cuda-12040, cuda-12030, cuda-12020, cuda-12010, cuda-12000, cuda-11080, cuda-11070, cuda-11060, cuda-11050, cuda-11040]");

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
    dynamic_linking(major, minor);

    #[cfg(feature = "static-linking")]
    static_linking(major, minor);
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
        "13.0" => (13, 0),
        "12.9" => (12, 9),
        "12.8" => (12, 8),
        "12.6" => (12, 6),
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
        "11.4" => (11, 4),
        v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
    }
}

#[allow(unused)]
fn dynamic_linking(major: usize, minor: usize) {
    for path in link_searches(major, minor) {
        println!("cargo:rustc-link-search=native={}", path.display());
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
    #[cfg(feature = "cupti")]
    println!("cargo:rustc-link-lib=dylib=cupti");
    #[cfg(feature = "cusparse")]
    println!("cargo:rustc-link-lib=dylib=cusparse");
    #[cfg(feature = "cusolver")]
    println!("cargo:rustc-link-lib=dylib=cusolver");
    #[cfg(feature = "cusolvermg")]
    println!("cargo:rustc-link-lib=dylib=cusolverMg");
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=dylib=cudnn");
    #[cfg(feature = "runtime")]
    println!("cargo:rustc-link-lib=dylib=cudart");
    #[cfg(feature = "cufile")]
    {
        println!("cargo:rustc-link-lib=dylib=cufile");
        println!("cargo:rustc-link-lib=dylib=cufile_rdma");
    }
    #[cfg(feature = "nvtx")]
    println!("cargo:rustc-link-lib=dylib=nvToolsExt");
}

#[allow(unused)]
fn static_linking(major: usize, minor: usize) {
    for path in link_searches(major, minor) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=static:+whole-archive=stdc++");
    #[cfg(any(feature = "driver", feature = "runtime"))]
    {
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=static:+whole-archive=cudart_static");
    }
    #[cfg(feature = "nccl")]
    println!("cargo:rustc-link-lib=static:+whole-archive=nccl_static");
    #[cfg(feature = "nvrtc")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=nvrtc_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=nvptxcompiler_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=nvrtc-builtins_static");
    }
    #[cfg(any(
        feature = "curand",
        feature = "cublas",
        feature = "cublaslt",
        feature = "cusparse",
        feature = "cusolver"
    ))]
    println!("cargo:rustc-link-lib=static:+whole-archive=culibos");
    #[cfg(feature = "curand")]
    println!("cargo:rustc-link-lib=static:+whole-archive=curand_static");
    #[cfg(feature = "cublas")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cublas_static");
    #[cfg(any(feature = "cublas", feature = "cublaslt"))]
    println!("cargo:rustc-link-lib=static:+whole-archive=cublasLt_static");
    #[cfg(feature = "cupti")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cupti_static");
    #[cfg(feature = "cusparse")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cusparse_static");
    #[cfg(feature = "cusolver")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=cusolver_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=cusolver_lapack_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=cusolver_metis_static");
    }
    #[cfg(feature = "cusolvermg")]
    println!("cargo:rustc-link-lib=dylib=cusolverMg");
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cudnn");
    #[cfg(feature = "cufile")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=cufile_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=cufile_rdma_static");
    }
    #[cfg(feature = "nvtx")]
    println!("cargo:rustc-link-lib=dylib=nvToolsExt");
}

#[allow(unused)]
fn link_searches(major: usize, minor: usize) -> Vec<PathBuf> {
    let env_vars = TYPICAL_CUDA_PATH_ENV_VARS
        .iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .collect::<Vec<_>>();

    // When building in a Conda-like environment with dynamic linking, if no
    // CUDA path is supplied, then it is higly likely that, by defaulting our
    // linker search paths to the typical locations below, linker errors will
    // occur. Print a warning with some guidance.
    #[cfg(feature = "dynamic-linking")]
    if env_vars.is_empty() && std::env::var("CONDA_PREFIX").is_ok() {
        println!("cargo::warning=Detected $CONDA_PREFIX, but no CUDA path was set through one of: {TYPICAL_CUDA_PATH_ENV_VARS:?}. Linking to system CUDA libraries; linker errors may occur. To use CUDA installed via conda please ensure the environment contains all required dependencies (e.g. the \"cuda-driver-dev\") and retry building with CUDA_HOME=$CONDA_PREFIX.")
    }

    let typical_locations = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/Program Files/NVIDIA",
        "C:/CUDA",
        // See issue #260 & #409
        // TODO figure out how to handle all of these automatically
        "C:/Program Files/NVIDIA/CUDNN/v9.10",
        "C:/Program Files/NVIDIA/CUDNN/v9.9",
        "C:/Program Files/NVIDIA/CUDNN/v9.8",
        "C:/Program Files/NVIDIA/CUDNN/v9.7",
        "C:/Program Files/NVIDIA/CUDNN/v9.6",
        "C:/Program Files/NVIDIA/CUDNN/v9.5",
        "C:/Program Files/NVIDIA/CUDNN/v9.4",
        "C:/Program Files/NVIDIA/CUDNN/v9.3",
        "C:/Program Files/NVIDIA/CUDNN/v9.2",
        "C:/Program Files/NVIDIA/CUDNN/v9.1",
        "C:/Program Files/NVIDIA/CUDNN/v9.0",
    ];

    let possible_locations = if env_vars.is_empty() {
        typical_locations
            .into_iter()
            .map(Into::<String>::into)
            .collect()
    } else {
        env_vars
    };

    let mut candidates = Vec::new();
    for root in possible_locations.into_iter().map(Into::<PathBuf>::into) {
        candidates.extend(
            [
                "lib".into(),
                "lib/stubs".into(),
                "lib/x64".into(),
                "lib/Win32".into(),
                "lib/x86_64".into(),
                "lib/x86_64-linux-gnu".into(),
                "lib64".into(),
                "lib64/stubs".into(),
                "targets/x86_64-linux".into(),
                "targets/x86_64-linux/lib".into(),
                "targets/x86_64-linux/lib/stubs".into(),
                // see issue #260
                std::format!("lib/{major}.{minor}/x64"),
                // see issue #260
                std::format!("lib/{major}.{minor}/x86_64"),
            ]
            .iter()
            .map(|p| root.join(p))
            .filter(|p| p.is_dir()),
        )
    }

    candidates
}
