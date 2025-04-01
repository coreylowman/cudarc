use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::fs::File;
use std::path::Path;
use tar::Archive;
use xz2::read::XzDecoder;

fn extract_tar_gz(tarball_path: &Path, output_dir: &Path) -> Result<()> {
    let tarball = File::open(tarball_path)
        .context(format!("Failed to open tarball {}", tarball_path.display()))?;

    let decompressed = GzDecoder::new(tarball);
    let mut archive = Archive::new(decompressed);

    archive.unpack(output_dir).with_context(|| {
        format!(
            "Failed to unpack {} to {}",
            tarball_path.display(),
            output_dir.display()
        )
    })?;

    Ok(())
}

fn extract_tar_xz(tarball_path: &Path, output_dir: &Path) -> Result<()> {
    let tarball = File::open(tarball_path)
        .context(format!("Failed to open tarball {}", tarball_path.display()))?;

    let decompressed = XzDecoder::new(tarball);
    let mut archive = Archive::new(decompressed);

    archive.unpack(output_dir).with_context(|| {
        format!(
            "Failed to unpack {} to {}",
            tarball_path.display(),
            output_dir.display()
        )
    })?;

    Ok(())
}

pub(crate) fn extract_archive(archive_path: &Path, output_dir: &Path) -> Result<()> {
    match archive_path.extension().and_then(|s| s.to_str()) {
        Some("gz") => extract_tar_gz(archive_path, output_dir),
        Some("xz") => extract_tar_xz(archive_path, output_dir),
        Some("tgz") => extract_tar_gz(archive_path, output_dir),
        Some("txz") => extract_tar_xz(archive_path, output_dir),
        _ => Err(anyhow::anyhow!(
            "Unsupported archive format: {}",
            archive_path.display()
        )),
    }
}
