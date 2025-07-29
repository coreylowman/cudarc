#[allow(unused)]
use std::{fs::File, os::fd::AsRawFd, sync::Arc};

use crate::driver::{CudaStream, DevicePtr, DevicePtrMut, DeviceRepr};

pub use super::result::CufileError;

use super::{result, sys};

/// Handle for [CUfile api](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
///
/// This object is thread safe.
#[derive(Debug)]
pub struct Cufile {}

unsafe impl Send for Cufile {}
unsafe impl Sync for Cufile {}

impl Drop for Cufile {
    fn drop(&mut self) {
        result::driver_close().unwrap();
    }
}

impl Cufile {
    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriveropen)
    pub fn new() -> Result<Arc<Self>, CufileError> {
        result::driver_open()?;
        Ok(Arc::new(Self {}))
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledrivergetproperties)
    pub fn get_properties(&self) -> Result<sys::CUfileDrvProps, CufileError> {
        result::driver_get_properties()
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriversetpollmode-bool-poll-size-t-poll-threshold-size)
    pub fn set_poll_mode(&self, poll: bool, poll_threshold_size: usize) -> Result<(), CufileError> {
        result::driver_set_poll_mode(poll, poll_threshold_size)
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriversetmaxdirectiosize-size-t-max-direct-io-size)
    pub fn set_max_direct_io_size(&self, max_direct_io_size: usize) -> Result<(), CufileError> {
        result::driver_set_max_direct_io_size(max_direct_io_size)
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#size-t-max-cache-size)
    pub fn set_max_cache_size(&self, max_cache_size: usize) -> Result<(), CufileError> {
        result::driver_set_max_cache_size(max_cache_size)
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriversetmaxpinnedmemsize-size-t-max-pinned-memory-size)
    pub fn set_max_pinned_mem_size(&self, max_pinned_size: usize) -> Result<(), CufileError> {
        result::driver_set_max_pinned_mem_size(max_pinned_size)
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilestreamregister)
    pub fn stream_register(
        &self,
        stream: &crate::driver::CudaStream,
        flags: u32,
    ) -> Result<(), CufileError> {
        unsafe { result::stream_register(stream.cu_stream() as _, flags) }
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilestreamderegister)
    pub fn stream_deregister(&self, stream: &crate::driver::CudaStream) -> Result<(), CufileError> {
        unsafe { result::stream_deregister(stream.cu_stream() as _) }
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufregister)
    pub fn buf_register<T, Ptr: DevicePtr<T>>(
        &self,
        buf: &crate::driver::CudaSlice<T>,
    ) -> Result<(), CufileError> {
        unsafe { result::buf_register(buf.cu_device_ptr as _, buf.len, 0) }
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufderegister)
    pub fn buf_deregister<T>(&self, buf: &crate::driver::CudaSlice<T>) -> Result<(), CufileError> {
        unsafe { result::buf_deregister(buf.cu_device_ptr as _) }
    }
}

/// A wrapper around [sys::CUfileHandle_t], created by [Cufile::register()],
/// used with [CudaStream::memcpy_dtof] and [CudaStream::memcpy_ftod].
#[derive(Debug)]
pub struct FileHandle {
    #[allow(unused)]
    file: File,
    handle: sys::CUfileHandle_t,
    #[allow(unused)]
    driver: Arc<Cufile>,
}

unsafe impl Send for FileHandle {}
unsafe impl Sync for FileHandle {}

impl Drop for FileHandle {
    fn drop(&mut self) {
        unsafe { result::handle_deregister(self.handle) }.unwrap();
    }
}

impl Cufile {
    /// Registers the `file` with the cufile driver. Only supported on windows & linux.
    ///
    /// The returned object can be used with [CudaStream::memcpy_dtof] and [CudaStream::memcpy_ftod].
    ///
    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilehandleregister)
    pub fn register(self: &Arc<Self>, file: File) -> Result<FileHandle, CufileError> {
        if cfg!(not(any(target_os = "linux", target_os = "windows"))) {
            Err(CufileError::Cufile(
                sys::CUfileOpError::CU_FILE_PLATFORM_NOT_SUPPORTED,
            ))
        } else {
            #[cfg(target_os = "linux")]
            let descr = sys::CUfileDescr_t {
                type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
                handle: sys::CUfileDescr_t__bindgen_ty_1 {
                    fd: file.as_raw_fd(),
                },
                fs_ops: std::ptr::null(),
            };

            #[cfg(target_os = "windows")]
            let descr = sys::CUfileDescr_t {
                type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_WIN32,
                handle: sys::CUfileDescr_t__bindgen_ty_1 {
                    handle: file.as_raw_fd(),
                },
                fs_ops: std::ptr::null(),
            };

            // NOTE: placeholder, shouldn't ever reach this
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            let descr = Default::default();

            let handle = result::handle_register(&descr)?;

            Ok(FileHandle {
                file,
                handle,
                driver: self.clone(),
            })
        }
    }
}

impl FileHandle {
    /// Underlying cuda handle
    pub fn cu(&self) -> sys::CUfileHandle_t {
        self.handle
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }
}

impl CudaStream {
    /// Copy memory from a file into a destination buffer *on the device*.
    ///
    /// The return value of this is initialized with 0, and after the operation successfully finishes
    /// on the stream, it will contain a value other than 0. See the docs for possible values.
    ///
    /// Wrapper around [cuFileReadAsync](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilereadasync)
    pub fn memcpy_ftod<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        fh: &FileHandle,
        file_offset: i64,
        dst: &mut Dst,
    ) -> Result<Box<isize>, CufileError> {
        let num_bytes = dst.num_bytes();
        let (dst, _record_dst) = dst.device_ptr_mut(self);
        let mut bytes_read = Box::new(0);
        unsafe {
            result::read_async(
                fh.cu() as _,
                dst as _,
                &num_bytes,
                &file_offset,
                &0,
                &mut bytes_read,
                self.cu_stream() as _,
            )
        }?;
        Ok(bytes_read)
    }

    /// Copy memory from a device buffer to a file.
    ///
    /// The return value of this is initialized with 0, and after the operation successfully finishes
    /// on the stream, it will contain a value other than 0. See the docs for possible values.
    ///
    /// Wrapper around [cuFileWriteAsync](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewriteasync)
    pub fn memcpy_dtof<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        fh: &mut FileHandle,
        file_offset: i64,
    ) -> Result<Box<isize>, CufileError> {
        let num_bytes = src.num_bytes();
        let (src, _record_src) = src.device_ptr(self);
        let mut bytes_written = Box::new(0);
        unsafe {
            result::write_async(
                fh.cu() as _,
                src as _,
                &num_bytes,
                &file_offset,
                &0,
                &mut bytes_written,
                self.cu_stream() as _,
            )
        }?;
        Ok(bytes_written)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Read, Seek},
    };

    use super::*;
    use crate::driver::*;

    #[test]
    fn test_dtof() -> Result<(), CufileError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cufile = Cufile::new()?;

        let file = std::fs::File::create("/tmp/cudarc-cufile-test_dtof").unwrap();
        let mut handle = cufile.register(file)?;

        let data = [0u8, 1, 2, 3, 4];
        let buf = stream.memcpy_stod(&data).unwrap();

        let written = stream.memcpy_dtof(&buf, &mut handle, 0)?;
        stream.synchronize().unwrap();
        assert_eq!(*written, data.len() as isize);

        handle.file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        handle.file.read_to_end(&mut buf).unwrap();
        assert_eq!(&buf, &data);

        Ok(())
    }

    #[test]
    fn test_ftod() -> Result<(), CufileError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cufile = Cufile::new()?;

        let data = [0u8, 1, 2, 3, 4];
        fs::write("/tmp/cudarc-cufile-test_ftod", &data).unwrap();

        let file = std::fs::File::open("/tmp/cudarc-cufile-test_ftod").unwrap();
        let mut handle = cufile.register(file)?;

        let mut buf = stream.alloc_zeros::<u8>(data.len()).unwrap();

        let read = stream.memcpy_ftod(&handle, 0, &mut buf)?;

        stream.synchronize().unwrap();
        assert_eq!(*read, data.len() as isize);

        // NOTE: asserting file is unchanged
        handle.file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        handle.file.read_to_end(&mut buf).unwrap();
        assert_eq!(&buf, &data);

        Ok(())
    }
}
