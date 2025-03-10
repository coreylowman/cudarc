use crate::driver::{
    result::{self, DriverError},
    sys,
};

use super::core::{CudaSlice, CudaStream};
use super::device_ptr::{DevicePtr, DevicePtrMut};
use super::host_slice::HostSlice;

use std::{marker::PhantomData, sync::Arc, vec::Vec};

/// Marker trait to indicate that the type is valid
/// when all of its bits are set to 0.
///
/// # Safety
/// Not all types are valid when all bits are set to 0.
/// Be very sure when implementing this trait!
pub unsafe trait ValidAsZeroBits {}
unsafe impl ValidAsZeroBits for bool {}
unsafe impl ValidAsZeroBits for i8 {}
unsafe impl ValidAsZeroBits for i16 {}
unsafe impl ValidAsZeroBits for i32 {}
unsafe impl ValidAsZeroBits for i64 {}
unsafe impl ValidAsZeroBits for i128 {}
unsafe impl ValidAsZeroBits for isize {}
unsafe impl ValidAsZeroBits for u8 {}
unsafe impl ValidAsZeroBits for u16 {}
unsafe impl ValidAsZeroBits for u32 {}
unsafe impl ValidAsZeroBits for u64 {}
unsafe impl ValidAsZeroBits for u128 {}
unsafe impl ValidAsZeroBits for usize {}
unsafe impl ValidAsZeroBits for f32 {}
unsafe impl ValidAsZeroBits for f64 {}
#[cfg(feature = "f16")]
unsafe impl ValidAsZeroBits for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl ValidAsZeroBits for half::bf16 {}
unsafe impl<T: ValidAsZeroBits, const M: usize> ValidAsZeroBits for [T; M] {}
/// Implement `ValidAsZeroBits` for tuples if all elements are `ValidAsZeroBits`,
///
/// # Note
/// This will also implement `ValidAsZeroBits` for a tuple with one element
macro_rules! impl_tuples {
    ($t:tt) => {
        impl_tuples!(@ $t);
    };
    // the $l is in front of the reptition to prevent parsing ambiguities
    ($l:tt $(,$t:tt)+) => {
        impl_tuples!($($t),+);
        impl_tuples!(@ $l $(,$t)+);
    };
    (@ $($t:tt),+) => {
        unsafe impl<$($t: ValidAsZeroBits,)+> ValidAsZeroBits for ($($t,)+) {}
    };
}
impl_tuples!(A, B, C, D, E, F, G, H, I, J, K, L);

/// Something that can be copied to device memory and
/// turned into a parameter for [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because a struct should likely
/// be `#[repr(C)]` to be represented in cuda memory,
/// and not all types are valid.
pub unsafe trait DeviceRepr {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

unsafe impl DeviceRepr for bool {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for i128 {}
unsafe impl DeviceRepr for isize {}
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for u128 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}
#[cfg(feature = "f16")]
unsafe impl DeviceRepr for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl DeviceRepr for half::bf16 {}

impl<T> CudaSlice<T> {
    /// Takes ownership of the underlying [sys::CUdeviceptr]. **It is up
    /// to the owner to free this value**.
    ///
    /// Drops the underlying host_buf if there is one.
    pub fn leak(self) -> sys::CUdeviceptr {
        let ptr = self.cu_device_ptr;
        std::mem::forget(self);
        ptr
    }
}

impl CudaStream {
    /// Creates a [CudaSlice] from a [sys::CUdeviceptr]. Useful in conjunction with
    /// [`CudaSlice::leak()`].
    ///
    /// # Safety
    /// - `cu_device_ptr` must be a valid allocation
    /// - `cu_device_ptr` must space for `len * std::mem::size_of<T>()` bytes
    /// - The memory may not be valid for type `T`, so some sort of memset operation
    ///   should be called on the memory.
    pub unsafe fn upgrade_device_ptr<T>(
        self: &Arc<Self>,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        let read = self.ctx.empty_event(None).unwrap();
        let write = self.ctx.empty_event(None).unwrap();
        CudaSlice {
            cu_device_ptr,
            len,
            read,
            write,
            stream: self.clone(),
            marker: PhantomData,
        }
    }
}

impl CudaStream {
    /// Allocates an empty [CudaSlice] with 0 length.
    pub fn null<T>(self: &Arc<Self>) -> Result<CudaSlice<T>, result::DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_device_ptr = if self.ctx.has_async_alloc {
            unsafe { result::malloc_async(self.cu_stream, 0) }?
        } else {
            unsafe { result::malloc_sync(0) }?
        };
        let read = self.ctx.empty_event(None)?;
        let write = self.ctx.empty_event(None)?;
        Ok(CudaSlice {
            cu_device_ptr,
            len: 0,
            read,
            write,
            stream: self.clone(),
            marker: PhantomData,
        })
    }

    /// # Safety
    /// This is unsafe because the memory is unset.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_device_ptr = if self.ctx.has_async_alloc {
            result::malloc_async(self.cu_stream, len * std::mem::size_of::<T>())?
        } else {
            result::malloc_sync(len * std::mem::size_of::<T>())?
        };
        let read = self.ctx.empty_event(None)?;
        let write = self.ctx.empty_event(None)?;
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            read,
            write,
            stream: self.clone(),
            marker: PhantomData,
        })
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(len) }?;
        self.memset_zeros(&mut dst)?;
        Ok(dst)
    }

    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        dst.block_for_write(self)?;
        unsafe {
            result::memset_d8_async(*dst.device_ptr_mut(), 0, dst.num_bytes(), self.cu_stream)
        }?;
        dst.record_write(self)?;
        Ok(())
    }

    /// Transfer a rust **s**lice to **d**evice
    pub fn memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        let src = unsafe { src.stream_synced_slice(self) }?;
        dst.block_for_write(self)?;
        unsafe { result::memcpy_htod_async(*dst.device_ptr_mut(), src, self.cu_stream) }?;
        src.record_use(self)?;
        dst.record_write(self)?;
        Ok(())
    }

    /// Transfer a **d**evice to rust **v**ec
    pub fn memcpy_dtov<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len())
        };
        self.memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        let dst = unsafe { dst.stream_synced_mut_slice(self) }?;
        assert!(dst.len() >= src.len());
        src.block_for_read(self)?;
        unsafe { result::memcpy_dtoh_async(dst, *src.device_ptr(), self.cu_stream) }?;
        src.record_read(self)?;
        dst.record_use(self)?;
        Ok(())
    }

    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        src.block_for_read(self)?;
        dst.block_for_write(self)?;
        unsafe {
            result::memcpy_dtod_async(
                *dst.device_ptr_mut(),
                *src.device_ptr(),
                src.num_bytes(),
                self.cu_stream,
            )
        }?;
        src.record_read(self)?;
        dst.record_write(self)?;
        Ok(())
    }

    pub fn clone_dtod<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::driver::CudaContext;

    use super::*;

    #[test]
    fn test_post_build_arc_count() {
        let ctx = CudaContext::new(0).unwrap();
        assert_eq!(Arc::strong_count(&ctx), 1);
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let ctx = CudaContext::new(0).unwrap();
        assert_eq!(Arc::strong_count(&ctx), 1);
        let stream = ctx.default_stream();
        assert_eq!(Arc::strong_count(&ctx), 2);
        let t = stream.alloc_zeros::<f32>(1).unwrap();
        assert_eq!(Arc::strong_count(&ctx), 3);
        drop(t);
        assert_eq!(Arc::strong_count(&ctx), 2);
        drop(stream);
        assert_eq!(Arc::strong_count(&ctx), 1);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let (free1, total1) = result::mem_get_info().unwrap();

        let t = stream.memcpy_stod(&[0.0f32; 5]).unwrap();
        let (free2, total2) = result::mem_get_info().unwrap();
        assert_eq!(total1, total2);
        assert!(free2 < free1);

        drop(t);
        ctx.synchronize().unwrap();

        let (free3, total3) = result::mem_get_info().unwrap();
        assert_eq!(total2, total3);
        assert!(free3 > free2);
        assert_eq!(free3, free1);
    }

    #[test]
    fn test_ctx_copy_to_views() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let smalls = [
            stream.memcpy_stod(&[-1.0f32, -0.8]).unwrap(),
            stream.memcpy_stod(&[-0.6, -0.4]).unwrap(),
            stream.memcpy_stod(&[-0.2, 0.0]).unwrap(),
            stream.memcpy_stod(&[0.2, 0.4]).unwrap(),
            stream.memcpy_stod(&[0.6, 0.8]).unwrap(),
        ];
        let mut big = stream.alloc_zeros::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.slice_mut(offset..offset + small.len());
            stream.memcpy_dtod(small, &mut sub).unwrap();
            offset += small.len();
        }

        assert_eq!(
            stream.memcpy_dtov(&big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }

    #[test]
    fn test_leak_and_upgrade() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let a = stream.memcpy_stod(&[1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let ptr = a.leak();
        let b = unsafe { stream.upgrade_device_ptr::<f32>(ptr, 3) };
        assert_eq!(stream.memcpy_dtov(&b).unwrap(), &[1.0, 2.0, 3.0]);

        let ptr = b.leak();
        let c = unsafe { stream.upgrade_device_ptr::<f32>(ptr, 5) };
        assert_eq!(stream.memcpy_dtov(&c).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// See https://github.com/coreylowman/cudarc/issues/160
    #[test]
    fn test_slice_is_freed_with_correct_context() {
        let ctx0 = CudaContext::new(0).unwrap();
        let slice = ctx0.default_stream().memcpy_stod(&[1.0; 10]).unwrap();
        let ctx1 = CudaContext::new(0).unwrap();
        ctx1.bind_to_thread().unwrap();
        drop(ctx0);
        drop(slice);
        drop(ctx1);
    }

    /// See https://github.com/coreylowman/cudarc/issues/161
    #[test]
    fn test_copy_uses_correct_context() {
        let ctx0 = CudaContext::new(0).unwrap();
        let _ctx1 = CudaContext::new(0).unwrap();
        let slice = ctx0.default_stream().memcpy_stod(&[1.0; 10]).unwrap();
        let _out = ctx0.default_stream().memcpy_dtov(&slice).unwrap();
    }

    #[test]
    fn test_htod_copy_pinned() {
        let truth = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut pinned = unsafe { ctx.alloc_pinned::<f32>(10) }.unwrap();
        pinned.as_mut_slice().unwrap().clone_from_slice(&truth);
        assert_eq!(pinned.as_slice().unwrap(), &truth);
        let dst = stream.memcpy_stod(&pinned).unwrap();
        let host = stream.memcpy_dtov(&dst).unwrap();
        assert_eq!(&host, &truth);
    }

    #[test]
    fn test_pinned_copy_is_faster() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let n = 100_000;
        let n_samples = 5;
        let not_pinned = std::vec![0.0f32; n];

        let start = Instant::now();
        for _ in 0..n_samples {
            let _ = stream.memcpy_stod(&not_pinned).unwrap();
            stream.synchronize().unwrap();
        }
        let unpinned_elapsed = start.elapsed() / n_samples;

        let pinned = unsafe { ctx.alloc_pinned::<f32>(n) }.unwrap();

        let start = Instant::now();
        for _ in 0..n_samples {
            let _ = stream.memcpy_stod(&pinned).unwrap();
            stream.synchronize().unwrap();
        }
        let pinned_elapsed = start.elapsed() / n_samples;

        // pinned memory transfer speed should be at least 2x faster, but this depends
        // on device
        assert!(
            pinned_elapsed.as_secs_f32() * 1.5 < unpinned_elapsed.as_secs_f32(),
            "{unpinned_elapsed:?} vs {pinned_elapsed:?}"
        );
    }
}
