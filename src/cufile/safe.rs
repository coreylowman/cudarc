use std::{boxed::Box, fs::File, pin::Pin, sync::Arc};

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
        // NOTE: we don't close this explicitly here because it happens on program close. Since this also
        // applies to ALL drivers open in the process, it can mess with multiple handles.
        // result::driver_close().unwrap();
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
    #[cfg(feature = "gte-12020")]
    pub fn stream_register(
        &self,
        stream: &crate::driver::CudaStream,
        flags: u32,
    ) -> Result<(), CufileError> {
        unsafe { result::stream_register(stream.cu_stream() as _, flags) }
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilestreamderegister)
    #[cfg(feature = "gte-12020")]
    pub fn stream_deregister(&self, stream: &crate::driver::CudaStream) -> Result<(), CufileError> {
        unsafe { result::stream_deregister(stream.cu_stream() as _) }
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufregister)
    pub fn buf_register<T>(&self, buf: &crate::driver::CudaSlice<T>) -> Result<(), CufileError> {
        unsafe { result::buf_register(buf.cu_device_ptr as _, buf.len, 0) }
    }

    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufderegister)
    pub fn buf_deregister<T>(&self, buf: &crate::driver::CudaSlice<T>) -> Result<(), CufileError> {
        unsafe { result::buf_deregister(buf.cu_device_ptr as _) }
    }
}

/// A wrapper around [sys::CUfileHandle_t], created by [Cufile::register()],
/// used with [FileHandle::sync_read], [FileHandle::sync_write], [CudaStream::memcpy_dtof], [CudaStream::memcpy_ftod].
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
    /// The returned object can be used with [FileHandle::sync_read], [FileHandle::sync_write], [CudaStream::memcpy_dtof], [CudaStream::memcpy_ftod].
    ///
    /// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilehandleregister)
    pub fn register(self: &Arc<Self>, file: File) -> Result<FileHandle, CufileError> {
        if cfg!(not(any(target_os = "linux", target_os = "windows"))) {
            Err(CufileError::Cufile(
                sys::CUfileOpError::CU_FILE_PLATFORM_NOT_SUPPORTED,
            ))
        } else {
            #[cfg(target_os = "linux")]
            let descr = {
                use std::os::unix::io::AsRawFd;
                sys::CUfileDescr_t {
                    type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
                    handle: sys::CUfileDescr_t__bindgen_ty_1 {
                        fd: file.as_raw_fd(),
                    },
                    fs_ops: std::ptr::null(),
                }
            };

            #[cfg(target_os = "windows")]
            let descr = {
                use std::os::windows::io::AsRawHandle;
                sys::CUfileDescr_t {
                    type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_WIN32,
                    handle: sys::CUfileDescr_t__bindgen_ty_1 {
                        handle: file.as_raw_handle(),
                    },
                    fs_ops: std::ptr::null(),
                }
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

impl FileHandle {
    /// Copy memory from a file into a destination buffer *on the device*.
    ///
    /// The return value of this is initialized with 0, and after the operation successfully finishes
    /// on the stream, it will contain a value other than 0. See the docs for possible values.
    ///
    /// Wrapper around [cuFileRead](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufileread).
    ///
    /// See [CudaStream::memcpy_ftod()] for async version.
    pub fn sync_read<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        &self,
        file_offset: i64,
        dst: &mut Dst,
    ) -> Result<isize, CufileError> {
        let stream = dst.stream().clone();
        let num_bytes = dst.num_bytes();
        let (dst, _record_dst) = dst.device_ptr_mut(&stream);
        stream.synchronize().unwrap();
        unsafe { result::read(self.cu(), dst as _, num_bytes, file_offset, 0) }
    }

    /// Copy memory from a device buffer to a file.
    ///
    /// The return value of this is initialized with 0, and after the operation successfully finishes
    /// on the stream, it will contain a value other than 0. See the docs for possible values.
    ///
    /// Wrapper around [cuFileWrite](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewrite).
    ///
    /// See [CudaStream::memcpy_dtof()] for async version.
    pub fn sync_write<T: DeviceRepr, Src: DevicePtr<T>>(
        &mut self,
        file_offset: i64,
        src: &Src,
    ) -> Result<isize, CufileError> {
        let stream = src.stream().clone();
        let num_bytes = src.num_bytes();
        let (src, _record_src) = src.device_ptr(&stream);
        stream.synchronize().unwrap();
        unsafe { result::write(self.cu(), src as _, num_bytes, file_offset, 0) }
    }
}

/// Result of [CudaStream::memcpy_dtof()]. Use [AsyncFileWrite::synchronize()] to get result of operation
#[allow(dead_code)]
#[derive(Debug)]
pub struct AsyncFileWrite<'a> {
    file_handle: &'a mut FileHandle,
    size: Pin<Box<usize>>,
    file_offset: Pin<Box<i64>>,
    buf_offset: Pin<Box<i64>>,
    bytes_written: Pin<Box<isize>>,
    event: crate::driver::CudaEvent,
}

impl<'a> AsyncFileWrite<'a> {
    /// Blocks host until operation completes & returns number of bytes written to file.
    pub fn synchronize(self) -> Result<isize, crate::driver::DriverError> {
        self.event.synchronize()?;
        Ok(*self.bytes_written)
    }
}

/// Result of [CudaStream::memcpy_ftod()]. Use [AsyncFileRead::synchronize()] to get result of operation
#[allow(dead_code)]
#[derive(Debug)]
pub struct AsyncFileRead<'a> {
    file_handle: &'a FileHandle,
    size: Pin<Box<usize>>,
    file_offset: Pin<Box<i64>>,
    buf_offset: Pin<Box<i64>>,
    bytes_read: Pin<Box<isize>>,
    event: crate::driver::CudaEvent,
}

impl<'a> AsyncFileRead<'a> {
    /// Blocks host until operation completes & returns number of bytes read from file.
    pub fn synchronize(self) -> Result<isize, crate::driver::DriverError> {
        self.event.synchronize()?;
        Ok(*self.bytes_read)
    }
}

impl CudaStream {
    /// Copy memory from a file into a destination buffer *on the device*.
    ///
    /// The return value of this is initialized with 0, and after the operation successfully finishes
    /// on the stream, it will contain a value other than 0. See the docs for possible values.
    ///
    /// Wrapper around [cuFileReadAsync](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilereadasync)
    ///
    /// See [FileHandle::sync_read()] for synchronous version.
    #[cfg(feature = "gte-12020")]
    pub fn memcpy_ftod<'a, T: DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        fh: &'a FileHandle,
        file_offset: i64,
        dst: &mut Dst,
    ) -> Result<AsyncFileRead<'a>, CufileError> {
        let event = self
            .ctx
            .new_event(Some(
                crate::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            ))
            .unwrap();

        let mut op = AsyncFileRead {
            file_handle: fh,
            size: Pin::new(Box::new(dst.num_bytes())),
            file_offset: Pin::new(Box::new(file_offset)),
            buf_offset: Pin::new(Box::new(0)),
            bytes_read: Pin::new(Box::new(0)),
            event,
        };

        let (dst, _record_dst) = dst.device_ptr_mut(self);

        unsafe {
            result::read_async(
                op.file_handle.cu(),
                dst as _,
                Pin::get_ref(op.size.as_ref()),
                Pin::get_ref(op.file_offset.as_ref()),
                Pin::get_ref(op.buf_offset.as_ref()),
                Pin::get_mut(op.bytes_read.as_mut()),
                self.cu_stream() as _,
            )
        }?;

        op.event.record(self).unwrap();

        Ok(op)
    }

    /// Copy memory from a device buffer to a file.
    ///
    /// The return value of this is initialized with 0, and after the operation successfully finishes
    /// on the stream, it will contain a value other than 0. See the docs for possible values.
    ///
    /// Wrapper around [cuFileWriteAsync](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewriteasync)
    ///
    /// See [FileHandle::sync_write()] for synchronous version.
    #[cfg(feature = "gte-12020")]
    pub fn memcpy_dtof<'a, T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        fh: &'a mut FileHandle,
        file_offset: i64,
    ) -> Result<AsyncFileWrite<'a>, CufileError> {
        let event = self
            .ctx
            .new_event(Some(
                crate::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            ))
            .unwrap();

        let mut op = AsyncFileWrite {
            file_handle: fh,
            size: Pin::new(Box::new(src.num_bytes())),
            file_offset: Pin::new(Box::new(file_offset)),
            buf_offset: Pin::new(Box::new(0)),
            bytes_written: Pin::new(Box::new(0)),
            event,
        };

        let (src, _record_src) = src.device_ptr(self);

        unsafe {
            result::write_async(
                op.file_handle.cu(),
                src as _,
                Pin::get_ref(op.size.as_ref()),
                Pin::get_ref(op.file_offset.as_ref()),
                Pin::get_ref(op.buf_offset.as_ref()),
                Pin::get_mut(op.bytes_written.as_mut()),
                self.cu_stream() as _,
            )
        }?;

        op.event.record(self).unwrap();

        Ok(op)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use std::{fs, vec::Vec};

    use super::*;
    use crate::driver::*;

    #[test]
    fn test_cufile_sync_dtof() -> Result<(), CufileError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let cufile = Cufile::new()?;

        let file = std::fs::File::create("/tmp/cudarc-cufile-test_dtof_sync").unwrap();
        let mut handle = cufile.register(file)?;

        let data = [0u8, 1, 2, 3, 4];
        let buf = stream.memcpy_stod(&data).unwrap();
        let written = handle.sync_write(0, &buf)?;
        assert_eq!(written, data.len() as isize);

        let buf = std::fs::read("/tmp/cudarc-cufile-test_dtof_sync").unwrap();
        assert_eq!(&buf, &data);

        Ok(())
    }

    #[test]
    fn test_cufile_sync_ftod() -> Result<(), CufileError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let cufile = Cufile::new()?;

        let data = [0u8, 1, 2, 3, 4];
        fs::write("/tmp/cudarc-cufile-test_ftod_sync", data).unwrap();

        let file = std::fs::File::open("/tmp/cudarc-cufile-test_ftod_sync").unwrap();
        let handle = cufile.register(file)?;
        let mut buf = stream.alloc_zeros::<u8>(data.len()).unwrap();
        let read = handle.sync_read(0, &mut buf)?;
        assert_eq!(read, data.len() as isize);

        // NOTE: asserting device equals our data
        let host_buf = stream.memcpy_dtov(&buf).unwrap();
        assert_eq!(&host_buf, &data);

        // NOTE: asserting file is unchanged
        let buf = fs::read("/tmp/cudarc-cufile-test_ftod_sync").unwrap();
        assert_eq!(&buf, &data);

        Ok(())
    }

    #[cfg(feature = "gte-12020")]
    #[test]
    fn test_cufile_async_dtof() -> Result<(), CufileError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let mut data: Vec<u8> = Vec::new();
        for i in 0..(1024 * 1024) {
            data.push((i % 256) as u8);
        }
        let buf = stream.memcpy_stod(&data).unwrap();

        let cufile = Cufile::new()?;
        let file = std::fs::File::create("/tmp/cudarc-cufile-test_dtof_async").unwrap();
        let mut handle = cufile.register(file)?;
        let write_op = stream.memcpy_dtof(&buf, &mut handle, 0).unwrap();
        let written = write_op.synchronize().unwrap();
        assert_eq!(written, data.len() as isize);

        let buf = std::fs::read("/tmp/cudarc-cufile-test_dtof_async").unwrap();
        assert_eq!(&buf, &data);

        Ok(())
    }

    #[cfg(feature = "gte-12020")]
    #[test]
    fn test_cufile_async_ftod() -> Result<(), CufileError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let cufile = Cufile::new()?;

        let mut data: Vec<u8> = Vec::new();
        for i in 0..(1024 * 1024) {
            data.push((i % 256) as u8);
        }
        fs::write("/tmp/cudarc-cufile-test_ftod_async", &data).unwrap();

        let file = std::fs::File::open("/tmp/cudarc-cufile-test_ftod_async").unwrap();
        let handle = cufile.register(file)?;

        let mut buf = stream.alloc_zeros::<u8>(data.len()).unwrap();
        let read_op = stream.memcpy_ftod(&handle, 0, &mut buf).unwrap();
        let read = read_op.synchronize().unwrap();
        assert_eq!(read, data.len() as isize);

        // NOTE: asserting device equals our data
        let host_buf = stream.memcpy_dtov(&buf).unwrap();
        assert_eq!(&host_buf, &data);

        // NOTE: asserting file is unchanged
        let buf = std::fs::read("/tmp/cudarc-cufile-test_ftod_async").unwrap();
        assert_eq!(&buf, &data);

        Ok(())
    }
}
