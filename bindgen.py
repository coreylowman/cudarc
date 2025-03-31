import os
import hashlib
import json
import functools
import subprocess
import urllib.request
from contextlib import chdir

cuda_versions = [
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
]

modules = {
    "cublas": {
        "cuda": "libcublas",
        "filters": {
            "types": ["^cublas.*"],
            "functions": ["^cublas.*"],
            "vars": ["^cublas.*"],
        },
        "libs": ["cublas"],
    },
    "cublaslt": {
        "cuda": "libcublas",
        "filters": {
            "types": ["^cublasLt.*"],
            "functions": ["^cublasLt.*"],
            "vars": ["^cublasLt.*"],
        },
        "libs": ["cublasLt"],
    },
    "curand": {
        "cuda": "libcurand",
        "filters": {
            "types": ["^curand.*"],
            "functions": ["^curand.*"],
            "vars": ["^curand.*"],
        },
        "libs": ["curand"],
    },
    "driver": {
        "cuda": "cuda_cudart",
        "filters": {
            "types": [
                "^CU.*",
                "^cuuint(32|64)_t",
                "^cudaError_enum",
                "^cu.*Complex$",
                "^cuda.*",
                "^libraryPropertyType.*",
            ],
            "functions": ["^cu.*"],
            "vars": ["^CU.*"],
        },
        "libs": ["cuda", "nvcuda"],
    },
    "runtime": {
        "cuda": "cuda_cudart",
        "filters": {
            "types": [
                "^[Cc][Uu][Dd][Aa].*",
            ],
            "functions": ["^[Cc][Uu][Dd][Aa].*"],
            "vars": ["^[Cc][Uu][Dd][Aa].*"],
        },
        "libs": ["cudart"],
    },
    "nvrtc": {
        "cuda": "cuda_nvrtc",
        "filters": {
            "types": ["^nvrtc.*"],
            "functions": ["^nvrtc.*"],
            "vars": ["^nvrtc.*"],
        },
        "libs": ["nvrtc"],
    },
    # Those are special
    "cudnn": {
        "cuda": "cudnn",
        "filters": {
            "types": ["^cudnn.*"],
            "functions": ["^cudnn.*"],
            "vars": ["^cudnn.*"],
        },
        "libs": ["cudnn"],
        "redist": {
            "url": "https://developer.download.nvidia.com/compute/cudnn/redist/",
            "version": "9.8.0",
        },
    },
    "nccl": {
        "cuda": "libnccl",
        "filters": {
            "types": ["^nccl.*"],
            "functions": ["^nccl.*"],
            "vars": ["^nccl.*"],
        },
        "libs": ["nccl"],
        "redist": {
            "url": "https://developer.download.nvidia.com/compute/redist/nccl/",
            "version": "2.26.2",
        },
    },
}
extra_modules = {}

MOD_RS = """
#[cfg(feature = "dynamic-loading")]
mod loaded;
#[cfg(feature = "dynamic-loading")]
pub use loaded::*;

#[cfg(not(feature = "dynamic-loading"))]
mod linked;
#[cfg(not(feature = "dynamic-loading"))]
pub use linked::*;
"""

IMPORT_RS = """
#[cfg(feature = "{0}")]
mod {1};
#[cfg(feature = "{0}")]
pub use {1}::*;
"""

LOADING_RS = """
pub unsafe fn culib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let lib_names = %s;
        let choices: Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = Lib::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}

mod adapter;
pub use adapter::*;
"""


def get_import_content():
    snippets = [
        IMPORT_RS.format(cuda_version, cuda_version.replace("cuda-", "sys_")).strip()
        for cuda_version in cuda_versions
    ]
    return "\n".join(snippets)


def get_version(cuda_version: str) -> [int, int, int]:
    number = cuda_version.split("-")[-1]
    major = int(number[:2])
    minor = int(number[2:4])
    patch = int(number[4:])
    return major, minor, patch


def download_url(url, filename):
    print(f"Downloading {url} into {filename}")
    if os.path.exists(filename):
        return
    chunk_size = 1024 * 1024  # 1MB chunks
    with urllib.request.urlopen(url) as response:
        with open(filename, "wb") as file:
            # Read and write the response in chunks to avoid loading the whole file into memory
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                file.write(chunk)


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        # Read and update hash string value in chunks of 4K
        for byte_block in iter(lambda: file.read(4096 * 4096), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def download_url_checked(url, filename, checksum):
    download_url(url, filename)
    assert calculate_sha256(filename) == checksum


@functools.cache
def get_redistrib(major, minor, patch, base_url):
    # TODO figure out a way to get directly the correct patch maybe.
    # This is simple enough for now.
    for patch in reversed(range(10)):
        try:
            filename = f"redistrib_{major}.{minor}.{patch}.json"
            url = f"{base_url}/{filename}"
            print(url)

            outfilename = f"downloads/{filename}"
            download_url(url, outfilename)
            break
        except urllib.error.HTTPError as e:
            print(e)
            continue
    else:
        raise RuntimeError("Couldn't find a suitable patch")
    with open(outfilename, "r") as f:
        content = f.read()

    data = json.loads(content)
    return data


def generate_sys(cuda_version, module_name, cuda_name, filters, lib_names):
    major, minor, patch = get_version(cuda_version)
    URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
    data = get_redistrib(major, minor, patch, URL)
    lib = data[cuda_name]["linux-x86_64"]
    path = lib["relative_path"]
    checksum = lib["sha256"]
    url = f"{URL}/{path}"

    output_directory = f"downloads/{module_name}"
    archive_directory = (
        f"{output_directory}/{os.path.splitext(os.path.basename(path))[0]}"
    )
    if not os.path.exists(archive_directory):
        os.makedirs(archive_directory)
        outfilename = f"{output_directory}/{os.path.basename(path)}"
        download_url_checked(url, outfilename, checksum)

        subprocess.run(
            [
                "tar",
                "xvf",
                outfilename,
                "--directory",
                output_directory,
                "--skip-old-files",
            ]
        )

    create_system_folders(
        cuda_version, module_name, filters, lib_names, archive_directory
    )


def create_system_folders(
    cuda_version, module_name, filters, lib_names, archive_directory
):
    cuda_directory = f"{archive_directory}/include/"
    main_directory = f"{os.environ['CUDA_ROOT']}/include/"

    sysdir = f"src/{module_name}/sys"
    os.makedirs(sysdir, exist_ok=True)

    with open(f"{sysdir}/mod.rs", "w") as f:
        f.write(MOD_RS.strip())

    os.makedirs(f"{sysdir}/linked/", exist_ok=True)

    outfilename = f"{cuda_version.replace('cuda-', 'sys_')}.rs"
    process = [
        "bindgen",
        *[f'--allowlist-type="{filter_name}"' for filter_name in filters["types"]],
        *[f'--allowlist-var="{filter_name}"' for filter_name in filters["vars"]],
        *[
            f'--allowlist-function="{filter_name}"'
            for filter_name in filters["functions"]
        ],
        "--default-enum-style",
        "rust",
        "--no-doc-comments",
        "--with-derive-default",
        "--with-derive-eq",
        "--with-derive-hash",
        "--with-derive-ord",
        "--use-core",
        # "--dynamic-loading",
        # "Lib",
        f"{os.getcwd()}/{sysdir}/wrapper.h",
        "--output",
        f"{sysdir}/linked/{outfilename}",
        "--",
        f"-I{os.path.join(os.getcwd(), cuda_directory)}",
        f"-I{main_directory}",
    ]
    print(" ".join(process))
    subprocess.run(" ".join(process), check=True, shell=True)

    import_content = get_import_content()
    with open(f"{sysdir}/linked/mod.rs", "w") as f:
        f.write(import_content)

    # >tmp.rs
    os.makedirs(f"{sysdir}/loaded/", exist_ok=True)
    process = [
        "bindgen",
        *[f'--allowlist-type="{filter_name}"' for filter_name in filters["types"]],
        *[f'--allowlist-var="{filter_name}"' for filter_name in filters["vars"]],
        *[
            f'--allowlist-function="{filter_name}"'
            for filter_name in filters["functions"]
        ],
        "--default-enum-style",
        "rust",
        "--no-doc-comments",
        "--with-derive-default",
        "--with-derive-eq",
        "--with-derive-hash",
        "--with-derive-ord",
        "--use-core",
        "--dynamic-loading",
        "Lib",
        f"{os.getcwd()}/{sysdir}/wrapper.h",
        "--output",
        f"{sysdir}/loaded/{outfilename}",
        "--",
        f"-I{os.path.join(os.getcwd(), cuda_directory)}",
        f"-I{main_directory}",
    ]
    print(" ".join(process))
    subprocess.run(" ".join(process), check=True, shell=True)

    subprocess.run(
        [
            "helper/target/release/cudarc_helper",
            "--input",
            f"{sysdir}/linked/{outfilename}",
            "--output",
            f"{sysdir}/loaded/adapter.rs",
        ],
        check=True,
    )

    import_content += "\n"
    libnames = "[" + ", ".join(f'"{item}"' for item in lib_names) + "]"
    # Old python formatting for less clashes
    import_content += LOADING_RS % libnames

    with open(f"{sysdir}/loaded/mod.rs", "w") as f:
        f.write(import_content)


def build_helper():
    with chdir("helper"):
        subprocess.run(["cargo", "build", "--release"], check=True)


def generate_cudnn(cuda_version, module_name, cuda_name, filters, lib_names, redist):
    cuda_major, _, _ = get_version(cuda_version)
    URL = redist["url"]
    major, minor, patch = redist["version"].split(".")
    data = get_redistrib(major, minor, patch, URL)
    lib = data[cuda_name]["linux-x86_64"]
    if cuda_major == 11:
        lib = lib["cuda11"]
    elif cuda_major == 12:
        lib = lib["cuda12"]
    else:
        raise RuntimeError(f"Unknown cuda version {cuda_major}")
    path = lib["relative_path"]
    checksum = lib["sha256"]
    url = f"{URL}/{path}"

    output_directory = f"downloads/{module_name}"
    archive_directory = (
        f"{output_directory}/{os.path.splitext(os.path.basename(path))[0]}"
    )
    if not os.path.exists(archive_directory):
        os.makedirs(archive_directory)
        outfilename = f"{output_directory}/{os.path.basename(path)}"
        download_url_checked(url, outfilename, checksum)

        subprocess.run(
            [
                "tar",
                "xvf",
                outfilename,
                "--directory",
                output_directory,
                "--skip-old-files",
            ]
        )
    create_system_folders(
        cuda_version, module_name, filters, lib_names, archive_directory
    )


def generate_nccl(cuda_version, module_name, cuda_name, filters, lib_names, redist):
    URL = redist["url"]
    version = redist["version"]
    # major, minor, patch = redist["version"].split(".")

    path = f"v{version}/nccl_{version}-1+cuda12.8_x86_64.txz"
    url = f"{URL}/{path}"
    print(url)

    output_directory = f"downloads/{module_name}"
    os.makedirs(output_directory, exist_ok=True)
    archive_directory = (
        f"{output_directory}/{os.path.splitext(os.path.basename(path))[0]}"
    )
    if not os.path.exists(archive_directory) or True:
        # os.makedirs(archive_directory)
        outfilename = f"{output_directory}/{os.path.basename(path)}"
        print(url)
        download_url(url, outfilename)

        subprocess.run(
            [
                "tar",
                "xvf",
                outfilename,
                "--directory",
                output_directory,
                "--skip-old-files",
            ]
        )
    create_system_folders(
        cuda_version, module_name, filters, lib_names, archive_directory
    )


generate = {
    "cudnn": generate_cudnn,
    "nccl": generate_nccl,
}


def main():
    os.makedirs("downloads", exist_ok=True)

    build_helper()
    for cuda_version in cuda_versions:
        print(f"=========== Cuda {cuda_version} ====================")
        for module_name, cuda in modules.items():
            print(f"----  {module_name}   ----")
            cuda_name = cuda["cuda"]
            filters = cuda["filters"]
            lib_names = cuda["libs"]
            if "redist" in cuda:
                redist = cuda["redist"]
                generate[module_name](
                    cuda_version, module_name, cuda_name, filters, lib_names, redist
                )
            else:
                generate_sys(cuda_version, module_name, cuda_name, filters, lib_names)


if __name__ == "__main__":
    main()
