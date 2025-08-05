use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use reqwest::blocking::{Response, get};
use serde_json::Value;
use sha2::{Digest, Sha256};

lazy_static! {
    static ref DOWNLOAD_CACHE: Mutex<HashMap<String, PathBuf>> = Mutex::new(HashMap::new());
    static ref REVISION: Mutex<HashMap<(u32, u32, u32, String), PathBuf>> =
        Mutex::new(HashMap::new());
}

fn download_response(url: &str) -> Result<Response> {
    Ok(get(url).unwrap())
}

pub fn to_file(url: &str, dest: &Path, multi_progress: &MultiProgress) -> Result<()> {
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

pub fn to_file_with_checksum(
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

    to_file(url, dest, multi_progress)?;
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
        if chunk.starts_with(&format!("redistrib_{major}.{minor}"))
            && chunk.ends_with(".json")
            // NOTE: some of the versions have a 4th part, and are formatted differently.
            && chunk.chars().filter(|&c| c == '.').count() == 3
        {
            redist = Some(chunk);
        }
    }

    let filename =
        redist.expect("Expected a redistrib.json file for {major}.{minor}.{patch} at {base_url}");

    let url = format!("{}/{}", base_url, filename);
    log::debug!("Trying {}", url);

    let out_path = Path::new("downloads").join(&filename);

    if to_file(&url, &out_path, multi_progress).is_ok() {
        let mut lock = REVISION.lock().unwrap();
        lock.insert(
            (major, minor, patch, base_url.to_string()),
            out_path.clone(),
        );
        return Ok(out_path);
    }
    Err(anyhow::anyhow!("Couldn't find a suitable patch"))
}

pub fn cuda_redist(
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
