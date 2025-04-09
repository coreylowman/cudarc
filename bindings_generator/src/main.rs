use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result};
use bindgen::Builder;
use lazy_static::lazy_static;
use reqwest::blocking::{Response, get};
use serde_json::Value;
use sha2::{Digest, Sha256};

mod extract;
mod merge;

lazy_static! {
    static ref DOWNLOAD_CACHE: Mutex<HashMap<String, PathBuf>> = Mutex::new(HashMap::new());
    static ref REVISION: Mutex<HashMap<(u32, u32, u32, String), PathBuf>> =
        Mutex::new(HashMap::new());
}

/// The cuda versions we're building against.
/// Those are the feature names used in cudarc
const CUDA_VERSIONS: &[&str] = &[
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
];

/// Cuda is split in various modules in cudarc.
/// Those configs decide how to download and
/// export bindings with bindgen. See [`ModuleConfig`].
fn create_modules() -> Vec<(String, ModuleConfig)> {
    vec![
        (
            "driver".to_string(),
            ModuleConfig {
                cuda: "cuda_cudart".to_string(),
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
                blocklist: Filters::none(),
                libs: vec!["cuda".to_string(), "nvcuda".to_string()],
                redist: None,
            },
        ),
        (
            "cublas".to_string(),
            ModuleConfig {
                cuda: "libcublas".to_string(),
                allowlist: Filters {
                    types: vec!["^cublas.*".to_string()],
                    functions: vec!["^cublas.*".to_string()],
                    vars: vec!["^cublas.*".to_string()],
                },
                blocklist: Filters::none(),
                libs: vec!["cublas".to_string()],
                redist: None,
            },
        ),
        (
            "cublaslt".to_string(),
            ModuleConfig {
                cuda: "libcublas".to_string(),
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
                redist: None,
            },
        ),
        (
            "curand".to_string(),
            ModuleConfig {
                cuda: "libcurand".to_string(),
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
                redist: None,
            },
        ),
        (
            "runtime".to_string(),
            ModuleConfig {
                cuda: "cuda_cudart".to_string(),
                allowlist: Filters {
                    types: vec!["^[Cc][Uu][Dd][Aa].*".to_string()],
                    functions: vec!["^[Cc][Uu][Dd][Aa].*".to_string()],
                    vars: vec!["^[Cc][Uu][Dd][Aa].*".to_string()],
                },
                blocklist: Filters::none(),
                libs: vec!["cudart".to_string()],
                redist: None,
            },
        ),
        (
            "nvrtc".to_string(),
            ModuleConfig {
                cuda: "cuda_nvrtc".to_string(),
                allowlist: Filters {
                    types: vec!["^nvrtc.*".to_string()],
                    functions: vec!["^nvrtc.*".to_string()],
                    vars: vec!["^nvrtc.*".to_string()],
                },
                blocklist: Filters::none(),
                libs: vec!["nvrtc".to_string()],
                redist: None,
            },
        ),
        (
            "cudnn".to_string(),
            ModuleConfig {
                cuda: "cudnn".to_string(),
                allowlist: Filters {
                    types: vec!["^cudnn.*".to_string()],
                    functions: vec!["^cudnn.*".to_string()],
                    vars: vec!["^cudnn.*".to_string()],
                },
                blocklist: Filters::none(),
                libs: vec!["cudnn".to_string()],
                redist: Some(Redist {
                    url: "https://developer.download.nvidia.com/compute/cudnn/redist/".to_string(),
                    version: "9.8.0".to_string(),
                }),
            },
        ),
        (
            "nccl".to_string(),
            ModuleConfig {
                cuda: "libnccl".to_string(),
                allowlist: Filters {
                    types: vec!["^nccl.*".to_string()],
                    functions: vec!["^nccl.*".to_string()],
                    vars: vec!["^nccl.*".to_string()],
                },
                blocklist: Filters::none(),
                libs: vec!["nccl".to_string()],
                redist: Some(Redist {
                    url: "https://developer.download.nvidia.com/compute/redist/nccl/".to_string(),
                    version: "2.26.2".to_string(),
                }),
            },
        ),
        (
            "cusparse".to_string(),
            ModuleConfig {
                cuda: "libcusparse".to_string(),
                filters: Filters {
                    types: vec!["^cusparse.*".to_string()],
                    functions: vec!["^cusparse.*".to_string()],
                    vars: vec!["^cusparse.*".to_string()],
                },
                libs: vec!["cusparse".to_string()],
                redist: None,
            },
        ),
        (
            "cusolver".to_string(),
            ModuleConfig {
                cuda: "libcusolver".to_string(),
                filters: Filters {
                    types: vec!["^cusolver.*".to_string()],
                    functions: vec!["^cusolver.*".to_string()],
                    vars: vec!["^cusolver.*".to_string()],
                },
                libs: vec!["cusolver".to_string()],
                // redist in cusolver is dummy
                redist: Some(Redist {
                    url: "".to_string(),
                    version: "".to_string(),
                }),
            },
        ),
    ]
}

#[derive(Debug)]
struct ModuleConfig {
    /// The name of the library within cuda/redist
    cuda: String,
    /// The various filter used in bindgen to select
    /// the symbols we re-expose
    allowlist: Filters,
    blocklist: Filters,
    /// The various names used to look for symbols
    /// Those names are only used with the `dynamic-loading`
    /// feature.
    libs: Vec<String>,
    /// Some libraries (`cudnn` and `nccl` are external to
    /// cuda/redist, and therefore require both custom
    /// code and custom information to get the redistributable
    /// archives
    redist: Option<Redist>,
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

#[derive(Debug)]
struct Redist {
    url: String,
    version: String,
}

/// Downloads, unpacks and generate bindings for all modules.
fn create_bindings(modules: &[(String, ModuleConfig)]) -> Result<()> {
    let downloads_dir = Path::new("downloads");
    fs::create_dir_all(downloads_dir).context("Failed to create downloads directory")?;

    let multi_progress = MultiProgress::new();
    let overall_pb = multi_progress.add(ProgressBar::new(CUDA_VERSIONS.len() as u64));
    overall_pb.set_style(
        ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len} ({eta})")?, // .progress_chars("#>-"),
    );

    for (i, cuda_version) in CUDA_VERSIONS.iter().enumerate() {
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

        for (module_name, module) in modules {
            module_pb.set_message(format!("{module_name}"));

            match &module.redist {
                Some(redist) => match module_name.as_str() {
                    "cudnn" => generate_cudnn(
                        cuda_version,
                        module_name,
                        module,
                        redist,
                        &primary_archives,
                        &multi_progress,
                    )
                    .context(format!("Failed to generate cudnn for {}", cuda_version))?,
                    "nccl" => generate_nccl(
                        cuda_version,
                        module_name,
                        module,
                        redist,
                        &primary_archives,
                        &multi_progress,
                    )
                    .context(format!("Failed to generate nccl for {}", cuda_version))?,
                    "cusolver" => generate_cusolver(
                        cuda_version,
                        module_name,
                        module,
                        &primary_archives,
                        &multi_progress,
                    )
                    .context(format!("Failed to generate cusolver for {}", cuda_version))?,
                    _ => unreachable!("Unknown module with redist: {}", module_name),
                },
                None => {
                    generate_sys(
                        cuda_version,
                        module_name,
                        module,
                        &primary_archives,
                        &multi_progress,
                    )
                    .context(format!("Failed to generate sys for {}", cuda_version))?;
                }
            }
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

fn download_response(url: &str) -> Result<Response> {
    Ok(get(url).unwrap())
}

fn download_to_file(url: &str, dest: &Path, multi_progress: &MultiProgress) -> Result<()> {
    log::debug!("Downloading  to file {} to {}", url, dest.display());
    if dest.exists() {
        // Add to cache if file exists
        log::debug!("File exists, inserting into cache");
        let mut cache = DOWNLOAD_CACHE.lock().expect("To get lock");
        cache.insert(url.to_string(), dest.to_path_buf());
        log::debug!("File exists, inserted");
        return Ok(());
    }

    log::debug!("Downloading url {url}");
    let mut response = download_response(url).expect("Downloading error");
    log::debug!("Got response");
    let status = response.status();
    if !status.is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download {}: HTTP {}",
            url,
            status
        ));
    }

    // Create parent directories if needed
    log::debug!("Checking parent directories {}", dest.display());
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).context(format!(
            "Failed to create parent directory for {}",
            dest.display()
        ))?;
    }

    log::debug!("Creating file");
    let pb = multi_progress.add(ProgressBar::new(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {wide_bar} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?,
    );
    pb.set_message(format!("Downloading {}", url));
    if let Some(total) = response.content_length() {
        pb.set_length(total);
    }

    let mut file =
        File::create(dest).context(format!("Failed to create file {}", dest.display()))?;
    log::debug!("Copying content");

    let mut buffer = [0; 4096];
    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        pb.inc(bytes_read as u64);
    }
    log::debug!("Copied content");

    // Add to cache after successful download
    let mut cache = DOWNLOAD_CACHE
        .lock()
        .expect("Failed to acquire download cache lock");
    cache.insert(url.to_string(), dest.to_path_buf());

    Ok(())
}

fn calculate_sha256(file_path: &Path) -> Result<String> {
    log::debug!("Opening file {}", file_path.display());
    let mut file =
        File::open(file_path).context(format!("Failed to open {}", file_path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 4096];

    loop {
        let bytes_read = file.read(&mut buffer).expect("Read from buffer");
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn download_with_checksum(
    url: &str,
    dest: &Path,
    checksum: &str,
    multi_progress: &MultiProgress,
) -> Result<()> {
    // Check cache first
    {
        let cache = DOWNLOAD_CACHE
            .lock()
            .expect("Failed to acquire download cache lock");
        if cache.get(url) == Some(&dest.to_path_buf()) {
            log::debug!("Already downloaded (cached): {}", url);
            return Ok(());
        }
    }

    log::debug!("Not in cache");
    if dest.exists() {
        log::debug!("Destination exists, comparing checksum");
        let actual_checksum = calculate_sha256(dest).context(format!(
            "Failed to calculate checksum for {}",
            dest.display()
        ))?;
        log::debug!("Checksum there");
        if actual_checksum == checksum {
            // Add to cache if file exists and checksum matches
            let mut cache = DOWNLOAD_CACHE
                .lock()
                .expect("Failed to acquire download cache lock");
            cache.insert(url.to_string(), dest.to_path_buf());
            return Ok(());
        }
        // Remove invalid file if checksum doesn't match
        fs::remove_file(dest)
            .context(format!("Failed to remove invalid file {}", dest.display()))?;
    }

    download_to_file(url, dest, multi_progress)?;
    log::debug!("Download ok");
    let actual_checksum = calculate_sha256(dest).context(format!(
        "Failed to calculate checksum for {}",
        dest.display()
    ))?;
    if actual_checksum != checksum {
        fs::remove_file(dest).context(format!(
            "Failed to remove invalid download {}",
            dest.display()
        ))?;
        return Err(anyhow::anyhow!(
            "Checksum mismatch for {}: expected {}, got {}",
            dest.display(),
            checksum,
            actual_checksum
        ));
    }
    log::debug!("Checksum ok");

    Ok(())
}

fn get_redistrib_path(
    major: u32,
    minor: u32,
    patch: u32,
    base_url: &str,
    multi_progress: &MultiProgress,
) -> Result<PathBuf> {
    {
        let revision = REVISION.lock().unwrap();
        if let Some(out_path) = revision.get(&(major, minor, patch, base_url.to_string())) {
            log::debug!("Already downloaded redistrib {}", out_path.display());
            return Ok(out_path.to_path_buf());
        }
    }

    let response = get(base_url).context("Getting redistrib version")?;
    let response = response.error_for_status()?;
    let content = response.text()?;
    let mut redist = None;
    for chunk in content.split("'") {
        if chunk.starts_with(&format!("redistrib_{major}.{minor}")) && chunk.ends_with(".json") {
            redist = Some(chunk);
        }
    }

    let filename =
        redist.expect("Expected a redistrib.json file for {major}.{minor}.{patch} at {base_url}");

    let url = format!("{}/{}", base_url, filename);
    log::debug!("Trying {}", url);

    let out_path = Path::new("downloads").join(&filename);

    if download_to_file(&url, &out_path, multi_progress).is_ok() {
        let mut lock = REVISION.lock().unwrap();
        lock.insert(
            (major, minor, patch, base_url.to_string()),
            out_path.clone(),
        );
        return Ok(out_path);
    }
    Err(anyhow::anyhow!("Couldn't find a suitable patch"))
}
fn get_redistrib(
    major: u32,
    minor: u32,
    patch: u32,
    base_url: &str,
    multi_progress: &MultiProgress,
) -> Result<Value> {
    let out_path = get_redistrib_path(major, minor, patch, base_url, multi_progress)?;
    let content = fs::read_to_string(&out_path)
        .context(format!("Failed to read cached file {}", out_path.display()))?;
    serde_json::from_str(&content)
        .context(format!("Failed to parse JSON from {}", out_path.display()))
}

fn get_archive(
    cuda_version: &str,
    cuda_name: &str,
    module_name: &str,
    multi_progress: &MultiProgress,
) -> Result<PathBuf> {
    let (major, minor, patch) = get_version(cuda_version)?;
    let url = "https://developer.download.nvidia.com/compute/cuda/redist/";
    let data = get_redistrib(major, minor, patch, url, multi_progress)?;

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
        download_with_checksum(
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
    module_name: &str,
    module: &ModuleConfig,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let cuda_name = &module.cuda;

    let archive_dir = get_archive(cuda_version, cuda_name, module_name, multi_progress)?;

    create_system_folders(
        cuda_version,
        module_name,
        &module.allowlist,
        &module.blocklist,
        &archive_dir,
        primary_archives,
    )?;
    Ok(())
}

fn create_system_folders(
    cuda_version: &str,
    module_name: &str,
    allowlist: &Filters,
    blocklist: &Filters,
    archive_directory: &Path,
    primary_archives: &[PathBuf],
) -> Result<()> {
    let sysdir = Path::new(".").join("out").join(module_name).join("sys");
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

    for filter_name in allowlist.types.iter() {
        builder = builder.allowlist_type(filter_name);
    }
    for filter_name in allowlist.vars.iter() {
        builder = builder.allowlist_var(filter_name);
    }
    for filter_name in allowlist.functions.iter() {
        builder = builder.allowlist_function(filter_name);
    }
    for filter_name in blocklist.types.iter() {
        builder = builder.blocklist_type(filter_name);
    }
    for filter_name in blocklist.vars.iter() {
        builder = builder.blocklist_var(filter_name);
    }
    for filter_name in blocklist.functions.iter() {
        builder = builder.blocklist_function(filter_name);
    }

    let parent_sysdir = Path::new("..").join("src").join(module_name).join("sys");
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

fn generate_cudnn(
    cuda_version: &str,
    module_name: &str,
    module: &ModuleConfig,
    redist: &Redist,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let cuda_name = &module.cuda;
    let (cuda_major, _, _) = get_version(cuda_version)?;
    let url = &redist.url;
    let version_parts: Vec<&str> = redist.version.split('.').collect();
    let major = version_parts[0].parse().context(format!(
        "Failed to parse major version from {}",
        redist.version
    ))?;
    let minor = version_parts[1].parse().context(format!(
        "Failed to parse minor version from {}",
        redist.version
    ))?;
    let patch = version_parts[2].parse().context(format!(
        "Failed to parse patch version from {}",
        redist.version
    ))?;

    let data = get_redistrib(major, minor, patch, url, multi_progress)?;
    let lib = &data[cuda_name]["linux-x86_64"];
    let lib = match cuda_major {
        11 => &lib["cuda11"],
        12 => &lib["cuda12"],
        _ => return Err(anyhow::anyhow!("Unknown cuda version {}", cuda_major)),
    };

    let path = lib["relative_path"].as_str().context(format!(
        "Missing relative_path in redistrib data for {}",
        cuda_name
    ))?;
    let checksum = lib["sha256"].as_str().context(format!(
        "Missing sha256 in redistrib data for {}",
        cuda_name
    ))?;
    let url = format!("{}/{}", url, path);

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
        download_with_checksum(&url, &out_path, checksum, multi_progress)?;
        extract::extract_archive(&out_path, &output_dir, multi_progress)
            .context("Extracting archive")?;
    }

    create_system_folders(
        cuda_version,
        module_name,
        &module.allowlist,
        &module.blocklist,
        &archive_dir,
        primary_archives,
    )
}

fn generate_nccl(
    cuda_version: &str,
    module_name: &str,
    module: &ModuleConfig,
    redist: &Redist,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let url = &redist.url;
    let version = &redist.version;

    let path = format!("v{}/nccl_{}-1+cuda12.8_x86_64.txz", version, version);
    let full_url = format!("{}/{}", url, path);
    log::debug!("{}", full_url);

    let output_dir = Path::new("downloads").join(module_name);
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
        download_to_file(&full_url, &out_path, multi_progress)
            .context(format!("Failed to download {}", full_url))?;

        extract::extract_archive(&out_path, &output_dir, multi_progress)?;
    }
    assert!(archive_dir.exists());

    create_system_folders(
        cuda_version,
        module_name,
        &module.allowlist,
        &module.blocklist,
        &archive_dir,
        primary_archives,
    )
}

fn generate_cusolver(
    cuda_version: &str,
    module_name: &str,
    module: &ModuleConfig,
    primary_archives: &[PathBuf],
    multi_progress: &MultiProgress,
) -> Result<()> {
    let cuda_name = &module.cuda;
    let filters = &module.filters;

    let archive_dir = get_archive(cuda_version, cuda_name, module_name, multi_progress)?;

    // copy essential cublas and cusolver to the archive directory
    let archive_dir_cublas = get_archive(cuda_version, "libcublas", "cublas", multi_progress)?;
    let archive_dir_cusparse =
        get_archive(cuda_version, "libcusparse", "cusparse", multi_progress)?;
    fs::copy(
        archive_dir_cublas.join("include").join("cublas_v2.h"),
        archive_dir.join("include").join("cublas_v2.h"),
    )
    .context(format!("Failed to copy cublas_v2.h"))?;
    fs::copy(
        archive_dir_cublas.join("include").join("cublas_api.h"),
        archive_dir.join("include").join("cublas_api.h"),
    )
    .context(format!("Failed to copy cublas_api.h"))?;
    fs::copy(
        archive_dir_cusparse.join("include").join("cusparse.h"),
        archive_dir.join("include").join("cusparse.h"),
    )
    .context(format!("Failed to copy cusparse.h"))?;

    create_system_folders(
        cuda_version,
        module_name,
        filters,
        &archive_dir,
        primary_archives,
    )?;
    Ok(())
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
}

fn main() -> Result<()> {
    let args = Args::parse();
    let modules = create_modules();
    if !args.skip_bindings {
        create_bindings(&modules)?;
    }
    merge::merge_bindings(&modules)?;
    Ok(())
}
