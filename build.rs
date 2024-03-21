use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    link_cuda();
}

#[allow(unused)]
fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    #[cfg(feature = "cuda_auto_version")]
    {
        let toolkit_root = root_candidates()
            .find(|path| path.join("include").join("cuda.h").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "Unable to find `include/cuda.h` under any of: {:?}. Set the `CUDA_ROOT` environment variable to `$CUDA_ROOT/include/cuda.h` to override path.",
                    root_candidates().collect::<Vec<_>>()
                )
            });

        use std::{fs::File, io::Read};
        let mut header = File::open(toolkit_root.join("include").join("cuda.h")).unwrap();
        let mut contents = String::new();
        header.read_to_string(&mut contents).unwrap();

        let key = "CUDA_VERSION ";
        let start = key.len() + contents.find(key).unwrap();
        match contents[start..].lines().next().unwrap() {
            "12020" => println!("cargo:rustc-cfg=feature=\"cuda_12020\""),
            "12010" => println!("cargo:rustc-cfg=feature=\"cuda_12010\""),
            "12000" => println!("cargo:rustc-cfg=feature=\"cuda_12000\""),
            "11080" => println!("cargo:rustc-cfg=feature=\"cuda_11080\""),
            v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
        }
    }
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
