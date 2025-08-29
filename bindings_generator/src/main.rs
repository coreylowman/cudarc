use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use bindgen::Builder;

mod download;
mod extract;
mod merge;

/// Cuda is split in various modules in cudarc.
/// Those configs decide how to download and
/// export bindings with bindgen. See [`ModuleConfig`].
fn create_modules() -> Vec<ModuleConfig> {
    vec![
        ModuleConfig {
            cudarc_name: "driver".to_string(),
            redist_name: "cuda_cudart".to_string(),
            allowlist: Filters {
                types: vec![
                    "^CU.*".to_string(),
                    "^cuuint(32|64)_t".to_string(),
                    "^cudaError_enum".to_string(),
                    "^cu.*Complex$".to_string(),
                    "^cuda.*".to_string(),
                    "^libraryPropertyType.*".to_string(),
                ],
                functions: vec!["^cu.*".to_string()],
                vars: vec!["^CU.*".to_string()],
            },
            blocklist: Filters {
                // NOTE: See https://github.com/coreylowman/cudarc/issues/385
                types: vec!["^cuCheckpoint.*".to_string()],
                functions: vec![
                    "^cuCheckpoint.*".to_string(),
                    "cuDeviceGetNvSciSyncAttributes".to_string(),
                ],
                vars: vec![],
            },
            libs: vec!["cuda".to_string(), "nvcuda".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cublas".to_string(),
            redist_name: "libcublas".to_string(),
            allowlist: Filters {
                types: vec!["^cublas.*".to_string()],
                functions: vec!["^cublas.*".to_string()],
                vars: vec!["^cublas.*".to_string()],
            },
            blocklist: Filters::none(),
            libs: vec!["cublas".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cublaslt".to_string(),
            redist_name: "libcublas".to_string(),
            allowlist: Filters {
                types: vec!["^cublasLt.*".to_string()],
                functions: vec!["^cublasLt.*".to_string()],
                vars: vec!["^cublasLt.*".to_string()],
            },
            blocklist: Filters {
                types: vec![],
                functions: vec!["cublasLtDisableCpuInstructionsSetMask".to_string()],
                vars: vec![],
            },
            libs: vec!["cublasLt".to_string()],
        },
        ModuleConfig {
            cudarc_name: "curand".to_string(),
            redist_name: "libcurand".to_string(),
            allowlist: Filters {
                types: vec!["^curand.*".to_string()],
                functions: vec!["^curand.*".to_string()],
                vars: vec!["^curand.*".to_string()],
            },
            blocklist: Filters {
                types: vec![],
                functions: vec![
                    "curandGenerateBinomial".to_string(),
                    "curandGenerateBinomialMethod".to_string(),
                ],
                vars: vec![],
            },
            libs: vec!["curand".to_string()],
        },
        ModuleConfig {
            cudarc_name: "runtime".to_string(),
            redist_name: "cuda_cudart".to_string(),
            allowlist: Filters {
                types: vec!["^[Cc][Uu][Dd][Aa].*".to_string()],
                functions: vec!["^[Cc][Uu][Dd][Aa].*".to_string()],
                vars: vec!["^[Cc][Uu][Dd][Aa].*".to_string()],
            },
            blocklist: Filters {
                // NOTE: See https://github.com/coreylowman/cudarc/issues/397
                types: vec![],
                functions: vec!["cudaDeviceGetNvSciSyncAttributes".to_string()],
                vars: vec![],
            },
            libs: vec!["cudart".to_string()],
        },
        ModuleConfig {
            cudarc_name: "nvrtc".to_string(),
            redist_name: "cuda_nvrtc".to_string(),
            allowlist: Filters {
                types: vec!["^nvrtc.*".to_string()],
                functions: vec!["^nvrtc.*".to_string()],
                vars: vec!["^nvrtc.*".to_string()],
            },
            blocklist: Filters {
                // NOTE: see https://github.com/coreylowman/cudarc/pull/431
                types: vec![],
                functions: vec![
                    "nvrtcGetPCHCreateStatus".to_string(),
                    "nvrtcGetPCHHeapSize".to_string(),
                    "nvrtcGetPCHHeapSizeRequired".to_string(),
                    "nvrtcSetFlowCallback".to_string(),
                    "nvrtcSetPCHHeapSize".to_string(),
                ],
                vars: vec![],
            },
            libs: vec!["nvrtc".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cudnn".to_string(),
            redist_name: "cudnn".to_string(),
            allowlist: Filters {
                types: vec!["^cudnn.*".to_string()],
                functions: vec!["^cudnn.*".to_string()],
                vars: vec!["^cudnn.*".to_string()],
            },
            blocklist: Filters::none(),
            libs: vec!["cudnn".to_string()],
        },
        ModuleConfig {
            cudarc_name: "nccl".to_string(),
            redist_name: "libnccl".to_string(),
            allowlist: Filters {
                types: vec!["^nccl.*".to_string()],
                functions: vec!["^nccl.*".to_string()],
                vars: vec!["^nccl.*".to_string()],
            },
            blocklist: Filters::none(),
            libs: vec!["nccl".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cusparse".to_string(),
            redist_name: "libcusparse".to_string(),
            allowlist: Filters {
                types: vec!["^cusparse.*".to_string()],
                functions: vec!["^cusparse.*".to_string()],
                vars: vec!["^cusparse.*".to_string()],
            },
            blocklist: Filters {
                types: vec![],
                functions: vec![
                    "cusparseCbsric02_bufferSizeExt".into(),
                    "cusparseCbsrilu02_bufferSizeExt".into(),
                    "cusparseCbsrsm2_bufferSizeExt".into(),
                    "cusparseCbsrsv2_bufferSizeExt".into(),
                    "cusparseCcsr2gebsr_bufferSizeExt".into(),
                    "cusparseCcsric02_bufferSizeExt".into(),
                    "cusparseCcsrilu02_bufferSizeExt".into(),
                    "cusparseCgebsr2gebsc_bufferSizeExt".into(),
                    "cusparseCgebsr2gebsr_bufferSizeExt".into(),
                    "cusparseDbsric02_bufferSizeExt".into(),
                    "cusparseDbsrilu02_bufferSizeExt".into(),
                    "cusparseDbsrsm2_bufferSizeExt".into(),
                    "cusparseDbsrsv2_bufferSizeExt".into(),
                    "cusparseDcsr2gebsr_bufferSizeExt".into(),
                    "cusparseDcsric02_bufferSizeExt".into(),
                    "cusparseDcsrilu02_bufferSizeExt".into(),
                    "cusparseDgebsr2gebsc_bufferSizeExt".into(),
                    "cusparseDgebsr2gebsr_bufferSizeExt".into(),
                    "cusparseSbsric02_bufferSizeExt".into(),
                    "cusparseSbsrilu02_bufferSizeExt".into(),
                    "cusparseSbsrsm2_bufferSizeExt".into(),
                    "cusparseSbsrsv2_bufferSizeExt".into(),
                    "cusparseScsr2gebsr_bufferSizeExt".into(),
                    "cusparseScsric02_bufferSizeExt".into(),
                    "cusparseScsrilu02_bufferSizeExt".into(),
                    "cusparseSgebsr2gebsc_bufferSizeExt".into(),
                    "cusparseSgebsr2gebsr_bufferSizeExt".into(),
                    "cusparseXgebsr2csr".into(),
                    "cusparseZbsric02_bufferSizeExt".into(),
                    "cusparseZbsrilu02_bufferSizeExt".into(),
                    "cusparseZbsrsm2_bufferSizeExt".into(),
                    "cusparseZbsrsv2_bufferSizeExt".into(),
                    "cusparseZcsr2gebsr_bufferSizeExt".into(),
                    "cusparseZcsric02_bufferSizeExt".into(),
                    "cusparseZcsrilu02_bufferSizeExt".into(),
                    "cusparseZgebsr2gebsc_bufferSizeExt".into(),
                    "cusparseZgebsr2gebsr_bufferSizeExt".into(),
                ],
                vars: vec![],
            },
            libs: vec!["cusparse".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cusolver".to_string(),
            redist_name: "libcusolver".to_string(),
            allowlist: Filters {
                types: vec!["^cusolver.*".to_string()],
                functions: vec!["^cusolver.*".to_string()],
                vars: vec!["^cusolver.*".to_string()],
            },
            blocklist: Filters {
                types: vec!["^cusolverMg.*".to_string()],
                functions: vec![
                    "^cusolverMg.*".to_string(),
                    "^cusolverDnLogger.*".to_string(),
                ],
                vars: vec!["^cusolverMg.*".to_string()],
            },
            libs: vec!["cusolver".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cusolvermg".to_string(),
            redist_name: "libcusolver".to_string(),
            allowlist: Filters {
                types: vec!["^cusolverMg.*".to_string()],
                functions: vec!["^cusolverMg.*".to_string()],
                vars: vec!["^cusolverMg.*".to_string()],
            },
            blocklist: Filters::none(),
            libs: vec!["cusolverMg".to_string()],
        },
        ModuleConfig {
            cudarc_name: "cufile".to_string(),
            redist_name: "libcufile".to_string(),
            allowlist: Filters {
                types: vec!["^[Cc][Uu][Ff][Ii][Ll][Ee].*".to_string()],
                functions: vec!["^cuFile.*".to_string()],
                vars: vec![],
            },
            blocklist: Filters::none(),
            libs: vec!["cufile".to_string()],
        },
    ]
}

#[derive(Debug)]
struct ModuleConfig {
    /// Name of corresponding module in cudarc
    cudarc_name: String,
    /// The name of the library within cuda/redist
    redist_name: String,
    /// The various filter used in bindgen to select
    /// the symbols we re-expose
    allowlist: Filters,
    blocklist: Filters,
    /// The various names used to look for symbols
    /// Those names are only used with the `dynamic-loading`
    /// feature.
    libs: Vec<String>,
}

impl ModuleConfig {
    fn run_bindgen(
        &self,
        cuda_version: &str,
        archive_directory: &Path,
        primary_archives: &[PathBuf],
    ) -> Result<()> {
        let sysdir = Path::new(".")
            .join("out")
            .join(&self.cudarc_name)
            .join("sys");
        fs::create_dir_all(&sysdir)
            .context(format!("Failed to create directory {}", sysdir.display()))?;

        let linked_dir = sysdir.join("linked");
        fs::create_dir_all(&linked_dir).context(format!(
            "Failed to create directory {}",
            linked_dir.display()
        ))?;

        let outfilename = linked_dir.join(format!("{}.rs", cuda_version.replace("cuda-", "sys_")));

        // Generate linked bindings using bindgen library
        let mut builder = Builder::default()
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_default(true)
            .derive_eq(true)
            .derive_hash(true)
            .derive_ord(true)
            .generate_comments(false)
            .layout_tests(false)
            .use_core();

        for filter_name in self.allowlist.types.iter() {
            builder = builder.allowlist_type(filter_name);
        }
        for filter_name in self.allowlist.vars.iter() {
            builder = builder.allowlist_var(filter_name);
        }
        for filter_name in self.allowlist.functions.iter() {
            builder = builder.allowlist_function(filter_name);
        }
        for filter_name in self.blocklist.types.iter() {
            builder = builder.blocklist_type(filter_name);
        }
        for filter_name in self.blocklist.vars.iter() {
            builder = builder.blocklist_var(filter_name);
        }
        for filter_name in self.blocklist.functions.iter() {
            builder = builder.blocklist_function(filter_name);
        }

        let parent_sysdir = Path::new("..")
            .join("src")
            .join(&self.cudarc_name)
            .join("sys");
        let wrapper_h = parent_sysdir.join("wrapper.h");
        let cuda_directory = archive_directory.join("include");
        let primary_includes: Vec<_> = primary_archives
            .into_iter()
            .map(|c| c.join("include"))
            .collect();
        log::debug!("Include directories {}", cuda_directory.display());
        log::debug!(
            "Include primary directories {:?}",
            primary_includes
                .iter()
                .map(|p| p.display())
                .collect::<Vec<_>>()
        );
        builder = builder
            .header(wrapper_h.to_string_lossy())
            .clang_arg(format!("-I{}", cuda_directory.display()))
            // For cuda profiler which has a very simple consistent API
            .clang_arg(format!(
                "-I{}",
                std::env::current_dir()
                    .expect("Current directory")
                    .join("include")
                    .display()
            ));
        for include in primary_includes {
            builder = builder.clang_arg(format!("-I{}", include.display()));
        }

        let bindings = builder.generate().context(format!(
            "Failed to generate bindings for {}",
            wrapper_h.display()
        ))?;

        bindings.write_to_file(&outfilename).context(format!(
            "Failed to write bindings to {}",
            outfilename.display()
        ))?;
        log::debug!("Wrote linked bindings to {}", outfilename.display());

        Ok(())
    }
}

#[derive(Debug)]
/// Bindgen filters
struct Filters {
    types: Vec<String>,
    functions: Vec<String>,
    vars: Vec<String>,
}

impl Filters {
    fn none() -> Self {
        Self {
            types: vec![],
            functions: vec![],
            vars: vec![],
        }
    }
}

/// Downloads, unpacks and generate bindings for all modules.
fn create_bindings(modules: &[ModuleConfig], cuda_versions: &[&str]) -> Result<()> {
    let downloads_dir = Path::new("downloads");
    fs::create_dir_all(downloads_dir).context("Failed to create downloads directory")?;

    let multi_progress = MultiProgress::new();
    let overall_pb = multi_progress.add(ProgressBar::new(cuda_versions.len() as u64));
    overall_pb.set_style(
        ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len} ({eta})")?, // .progress_chars("#>-"),
    );

    for (i, cuda_version) in cuda_versions.iter().enumerate() {
        overall_pb.set_position(i as u64);
        overall_pb.set_message(format!("{}", cuda_version));

        let mut primary_archives = vec![];

        let names = if cuda_version.starts_with("cuda-12") {
            vec!["cuda_cudart", "cuda_nvcc", "cuda_cccl"]
        } else {
            vec!["cuda_cudart", "cuda_nvcc"]
        };

        let module_pb = multi_progress.add(ProgressBar::new((modules.len() + names.len()) as u64));

        module_pb.set_style(
            ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len} ({eta})")?,
        );
        for name in names {
            module_pb.set_message(format!("{name}"));
            let archive = get_archive(cuda_version, name, "primary", &multi_progress)?;
            primary_archives.push(archive);
            module_pb.inc(1);
        }

        for module in modules {
            module_pb.set_message(module.cudarc_name.clone());
            match module.cudarc_name.as_str() {
                "cudnn" => generate_cudnn(cuda_version, module, &primary_archives, &multi_progress),
                "nccl" => generate_nccl(cuda_version, module, &primary_archives, &multi_progress),
                _ => generate_sys(cuda_version, module, &primary_archives, &multi_progress),
            }
            .context(format!(
                "Failed to generate {} for {cuda_version}",
                module.cudarc_name
            ))?;
            module_pb.inc(1);
        }
        overall_pb.set_message(format!("Cuda version {cuda_version}"));
        overall_pb.inc(1);
    }

    overall_pb.finish_with_message("Completed all CUDA versions");
    Ok(())
}

fn get_version(cuda_version: &str) -> Result<(u32, u32, u32)> {
    let number = cuda_version
        .split('-')
        .last()
        .context(format!("Invalid CUDA version format: {}", cuda_version))?;

    let major = number[..2].parse().context(format!(
        "Failed to parse major version from {}",
        cuda_version
    ))?;
    let minor = number[2..4].parse().context(format!(
        "Failed to parse minor version from {}",
        cuda_version
    ))?;
    let patch = number[4..].parse().context(format!(
        "Failed to parse patch version from {}",
        cuda_version
    ))?;

    Ok((major, minor, patch))
}

fn get_archive(
    cuda_version: &str,
    cuda_name: &str,
    module_name: &str,
    multi_progress: &MultiProgress,
) -> Result<PathBuf> {
    let (major, minor, patch) = get_version(cuda_version)?;
    let url = "https://developer.download.nvidia.com/compute/cuda/redist/";
    let data = download::cuda_redist(major, minor, patch, url, multi_progress)?;

    let lib = &data[cuda_name]["linux-x86_64"];
    let path = lib["relative_path"].as_str().context(format!(
        "Missing relative_path in redistrib data for {}",
        cuda_name
    ))?;
    let checksum = lib["sha256"].as_str().context(format!(
        "Missing sha256 in redistrib data for {}",
        cuda_name
    ))?;

    let output_dir = Path::new("downloads").join(module_name);
    let parts: Vec<_> = Path::new(path)
        .file_name()
        .context(format!("Failed to get file name from {}", path))?
        .to_str()
        .expect("A valid filename")
        .split(".")
        .collect();
    let n = parts.len();
    let name = parts.into_iter().take(n - 2).collect::<Vec<_>>().join(".");
    let archive_dir = output_dir.join(name);
    log::debug!("Archive dir {archive_dir:?}");

    if !archive_dir.exists() {
        fs::create_dir_all(&archive_dir).context(format!(
            "Failed to create directory {}",
            archive_dir.display()
        ))?;
        let out_path = output_dir.join(
            Path::new(path)
                .file_name()
                .context(format!("Failed to get file name from {}", path))?,
        );
        log::debug!("Getting with checksum {url}/{path}");
        download::to_file_with_checksum(
            &format!("{}/{}", url, path),
            &out_path,
            checksum,
            multi_progress,
        )?;
        log::debug!("Got with checksum {url}/{path}");

        log::debug!("Extracting {}", out_path.display());
        extract::extract_archive(&out_path, &output_dir, multi_progress)?;
        log::debug!("Extracted {}", out_path.display());
    }
    Ok(archive_dir)
}

fn generate_sys(
    cuda_version: &str,
    module: &ModuleConfig,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let archive_dir = get_archive(
        cuda_version,
        &module.redist_name,
        &module.cudarc_name,
        multi_progress,
    )?;
    module.run_bindgen(cuda_version, &archive_dir, primary_archives)?;
    Ok(())
}

fn generate_cudnn(
    cuda_version: &str,
    module: &ModuleConfig,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let url = "https://developer.download.nvidia.com/compute/cudnn/redist/";

    let cuda_name = &module.redist_name;
    let (cuda_major, _, _) = get_version(cuda_version)?;

    let (major, minor, patch) = match cuda_major {
        11 => (9, 10, 2), // NOTE: this is the last cudnn version that supports cuda 11
        12 => (9, 12, 0),
        13 => (9, 12, 0),
        _ => return Err(anyhow::anyhow!("Unknown cuda version {}", cuda_major)),
    };

    let data = download::cuda_redist(major, minor, patch, url, multi_progress)?;
    let lib = &data[cuda_name]["linux-x86_64"];
    let lib = match cuda_major {
        11 => &lib["cuda11"],
        12 => &lib["cuda12"],
        13 => &lib["cuda13"],
        _ => return Err(anyhow::anyhow!("Unknown cuda version {}", cuda_major)),
    };

    let path = lib["relative_path"].as_str().context(format!(
        "Missing relative_path in redistrib data for {cuda_name}",
    ))?;
    let checksum = lib["sha256"]
        .as_str()
        .context(format!("Missing sha256 in redistrib data for {cuda_name}"))?;
    let url = format!("{url}/{path}");

    let output_dir = Path::new("downloads").join(&module.cudarc_name);
    let parts: Vec<_> = Path::new(path)
        .file_name()
        .context(format!("Failed to get file name from {path}"))?
        .to_str()
        .expect("A valid filename")
        .split(".")
        .collect();
    let n = parts.len();
    let name = parts.into_iter().take(n - 2).collect::<Vec<_>>().join(".");
    let archive_dir = output_dir.join(name);

    if !archive_dir.exists() {
        fs::create_dir_all(&archive_dir).context(format!(
            "Failed to create directory {}",
            archive_dir.display()
        ))?;
        let out_path = output_dir.join(
            Path::new(path)
                .file_name()
                .context(format!("Failed to get file name from {path}"))?,
        );
        download::to_file_with_checksum(&url, &out_path, checksum, multi_progress)?;
        extract::extract_archive(&out_path, &output_dir, multi_progress)
            .context("Extracting archive")?;
    }

    module.run_bindgen(cuda_version, &archive_dir, primary_archives)
}

fn generate_nccl(
    cuda_version: &str,
    module: &ModuleConfig,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let url = "https://developer.download.nvidia.com/compute/redist/nccl/";
    let version = "2.26.2";

    let path = format!("v{version}/nccl_{version}-1+cuda12.8_x86_64.txz");
    let full_url = format!("{url}/{path}");
    log::debug!("{}", full_url);

    let output_dir = Path::new("downloads").join(&module.cudarc_name);
    fs::create_dir_all(&output_dir).context(format!(
        "Failed to create directory {}",
        output_dir.display()
    ))?;
    let parts: Vec<_> = Path::new(&path)
        .file_name()
        .context(format!("Failed to get file name from {}", path))?
        .to_str()
        .expect("A valid filename")
        .split(".")
        .collect();
    let n = parts.len();
    // XXX: Extension is not .tar.gz but .txz
    let name = parts.into_iter().take(n - 1).collect::<Vec<_>>().join(".");
    let archive_dir = output_dir.join(name);

    if !archive_dir.exists() {
        let out_path = output_dir.join(
            Path::new(&path)
                .file_name()
                .context(format!("Failed to get file name from {}", path))?,
        );
        download::to_file(&full_url, &out_path, multi_progress)
            .context(format!("Failed to download {}", full_url))?;

        extract::extract_archive(&out_path, &output_dir, multi_progress)?;
    }
    assert!(archive_dir.exists());

    module.run_bindgen(cuda_version, &archive_dir, primary_archives)
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Generating the bindings from scratch takes a long
    /// time, but even if every archive is there too
    /// because we have to check Nvidia's website for updates
    /// Using this flag will skip that steps if you know you bindings
    /// exist and are up to date.
    #[arg(long, action)]
    skip_bindings: bool,

    #[arg(long, action)]
    version: Option<String>,

    /// Specify a single target to generate bindings for.
    #[arg(long, action)]
    target: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut modules = create_modules();
    if let Some(target) = args.target {
        modules.retain(|m| m.cudarc_name.contains(&target));
    }

    let mut cuda_versions = vec![
        "cuda-11040",
        "cuda-11050",
        "cuda-11060",
        "cuda-11070",
        "cuda-11080",
        "cuda-12000",
        "cuda-12010",
        "cuda-12020",
        "cuda-12030",
        "cuda-12040",
        "cuda-12050",
        "cuda-12060",
        "cuda-12080",
        "cuda-12090",
        "cuda-13000",
    ];
    if let Some(version) = args.version {
        cuda_versions.retain(|&v| v == version);
    }

    if !args.skip_bindings {
        create_bindings(&modules, &cuda_versions)?;
    }
    merge::merge_bindings(&modules)?;

    std::process::Command::new("cargo")
        .arg("fmt")
        .current_dir(std::fs::canonicalize("../")?)
        .status()?;
    Ok(())
}
