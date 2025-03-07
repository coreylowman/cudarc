use crate::driver::{
    result::{self, DriverError},
    sys,
};

use super::device_ptr::{DevicePtr, DevicePtrMut};
use super::host_slice::PinnedHostSlice;
use super::{
    core::{CudaDevice, CudaSlice, CudaStream},
    HostSlice,
};

use core::marker::PhantomData;
use std::{marker::Unpin, sync::Arc, vec::Vec};

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

impl CudaDevice {
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
        self.stream.upgrade_device_ptr(cu_device_ptr, len)
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

impl CudaDevice {
    /// Allocates an empty [CudaSlice] with 0 length.
    pub fn null<T>(self: &Arc<Self>) -> Result<CudaSlice<T>, result::DriverError> {
        self.stream.null()
    }

    /// Allocates device memory and increments the reference counter of [CudaDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        self.stream.alloc(len)
    }

    /// Allocates page locked host memory with `sys::CU_MEMHOSTALLOC_WRITECOMBINED` flags.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)
    ///
    /// # Safety
    /// 1. This is unsafe because the memory is unset after this call.
    pub unsafe fn alloc_pinned<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<PinnedHostSlice<T>, result::DriverError> {
        self.stream.ctx.alloc_pinned(len)
    }

    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn alloc_zeros<T: ValidAsZeroBits + DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        self.stream.alloc_zeros(len)
    }

    /// Sets all memory to 0 asynchronously.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn memset_zeros<T: ValidAsZeroBits + DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        self.stream.memset_zeros(dst)
    }

    /// Device to device copy (safe version of [result::memcpy_dtod_async]).
    ///
    /// # Panics
    ///
    /// If the length of the two values are different
    ///
    /// # Safety
    /// 1. We are guarunteed that `src` and `dst` are pointers to the same underlying
    ///     type `T`
    /// 2. Since they are both references, they can't have been freed
    /// 3. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn dtod_copy<T: DeviceRepr, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        self.stream.memcpy_dtod(src, dst)
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy<T: Unpin + DeviceRepr>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        self.stream.memcpy_stod(&src)
    }

    /// Asynchronously copies pinned host data (allocated with [CudaDevice::alloc_pinned()] to the device.
    ///
    /// This is generally at least 2x as fast as synchronous copies.
    ///
    /// # Safety
    /// 1. We don't have to synchronize at all here because the cuda driver allocated the [PinnedHostSlice].
    pub fn htod_copy_pinned<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        src: &PinnedHostSlice<T>,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.stream.alloc(src.len()) }?;
        self.stream.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy_into<T: DeviceRepr + Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), result::DriverError> {
        self.stream.memcpy_htod(&src, dst)
    }

    /// Allocates new device memory and synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::htod_copy()].
    ///
    /// # Safety
    ///
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_sync_copy<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &[T],
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.stream.alloc(src.len()) }?;
        self.stream.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::htod_copy()].
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_sync_copy_into<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        self.stream.memcpy_htod(src, dst)
    }

    /// Synchronously copies device memory into host memory.
    /// Unlike [`CudaDevice::dtoh_sync_copy_into`] this returns a [`Vec<T>`].
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` (after returning) it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    #[allow(clippy::uninit_vec)]
    pub fn dtoh_sync_copy<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, result::DriverError> {
        self.stream.memcpy_dtov(src)
    }

    /// Synchronously copies device memory into host memory
    ///
    /// Use [`CudaDevice::dtoh_sync_copy`] if you need [`Vec<T>`] and can't provide
    /// a correctly sized slice.
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn dtoh_sync_copy_into<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut [T],
    ) -> Result<(), result::DriverError> {
        self.stream.memcpy_dtoh(src, dst)
    }

    /// Synchronously de-allocates `src` and converts it into it's host value.
    /// You can just [drop] the slice if you don't need the host data.
    ///
    /// # Safety
    /// 1. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_reclaim<T: Clone + Default + DeviceRepr + Unpin>(
        self: &Arc<Self>,
        src: CudaSlice<T>,
    ) -> Result<Vec<T>, result::DriverError> {
        let mut dst = vec![Default::default(); src.len];
        self.stream.memcpy_dtoh(&src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronizes the stream.
    pub fn synchronize(self: &Arc<Self>) -> Result<(), result::DriverError> {
        self.stream.synchronize()
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
    pub fn memcpy_stod<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &[T],
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
        unsafe { dst.set_len(src.len()) };
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

    pub fn memcpy_dtod<T: DeviceRepr, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
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

    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::synchronize(self.cu_stream) }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::driver::DeviceSlice;

    use super::*;
    #[cfg(feature = "no-std")]
    use no_std_compat::vec;

    #[test]
    fn test_post_build_arc_count() {
        let device = CudaDevice::new(0).unwrap();
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.alloc_zeros::<f32>(1).unwrap();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
    }

    #[test]
    fn test_post_take_arc_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f32; 5].to_vec()).unwrap();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f64; 10].to_vec()).unwrap();
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_arc_slice_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = Arc::new(device.htod_copy::<f64>([0.0; 10].to_vec()).unwrap());
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_release_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([1.0f32, 2.0, 3.0].to_vec()).unwrap();
        #[allow(clippy::redundant_clone)]
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);

        let r_host = device.sync_reclaim(r).unwrap();
        assert_eq!(&r_host, &[1.0, 2.0, 3.0]);
        assert_eq!(Arc::strong_count(&device), 2);

        drop(r_host);
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let device = CudaDevice::new(0).unwrap();
        let (free1, total1) = result::mem_get_info().unwrap();

        let t = device.htod_copy([0.0f32; 5].to_vec()).unwrap();
        let (free2, total2) = result::mem_get_info().unwrap();
        assert_eq!(total1, total2);
        assert!(free2 < free1);

        drop(t);
        device.synchronize().unwrap();

        let (free3, total3) = result::mem_get_info().unwrap();
        assert_eq!(total2, total3);
        assert!(free3 > free2);
        assert_eq!(free3, free1);
    }

    #[test]
    fn test_device_copy_to_views() {
        let dev = CudaDevice::new(0).unwrap();

        let smalls = [
            dev.htod_copy(std::vec![-1.0f32, -0.8]).unwrap(),
            dev.htod_copy(std::vec![-0.6, -0.4]).unwrap(),
            dev.htod_copy(std::vec![-0.2, 0.0]).unwrap(),
            dev.htod_copy(std::vec![0.2, 0.4]).unwrap(),
            dev.htod_copy(std::vec![0.6, 0.8]).unwrap(),
        ];
        let mut big = dev.alloc_zeros::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.try_slice_mut(offset..offset + small.len()).unwrap();
            dev.dtod_copy(small, &mut sub).unwrap();
            offset += small.len();
        }

        assert_eq!(
            dev.sync_reclaim(big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }

    #[test]
    fn test_leak_and_upgrade() {
        let dev = CudaDevice::new(0).unwrap();

        let a = dev
            .htod_copy(std::vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
            .unwrap();

        let ptr = a.leak();
        let b = unsafe { dev.upgrade_device_ptr::<f32>(ptr, 3) };
        assert_eq!(dev.dtoh_sync_copy(&b).unwrap(), &[1.0, 2.0, 3.0]);

        let ptr = b.leak();
        let c = unsafe { dev.upgrade_device_ptr::<f32>(ptr, 5) };
        assert_eq!(dev.dtoh_sync_copy(&c).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// See https://github.com/coreylowman/cudarc/issues/160
    #[test]
    fn test_slice_is_freed_with_correct_context() {
        if CudaDevice::count().unwrap() < 2 {
            return;
        }
        let dev0 = CudaDevice::new(0).unwrap();
        let slice = dev0.htod_copy(vec![1.0; 10]).unwrap();
        let dev1 = CudaDevice::new(1).unwrap();
        drop(dev1);
        drop(dev0);
        drop(slice);
    }

    /// See https://github.com/coreylowman/cudarc/issues/161
    #[test]
    fn test_copy_uses_correct_context() {
        if CudaDevice::count().unwrap() < 2 {
            return;
        }
        let dev0 = CudaDevice::new(0).unwrap();
        let _dev1 = CudaDevice::new(1).unwrap();
        let slice = dev0.htod_copy(vec![1.0; 10]).unwrap();
        let _out = dev0.dtoh_sync_copy(&slice).unwrap();
    }

    #[test]
    fn test_htod_copy_pinned() {
        let truth = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let dev = CudaDevice::new(0).unwrap();
        let mut pinned = unsafe { dev.alloc_pinned::<f32>(10) }.unwrap();
        pinned.as_mut_slice().unwrap().clone_from_slice(&truth);
        assert_eq!(pinned.as_slice().unwrap(), &truth);
        let dst = dev.htod_copy_pinned(&pinned).unwrap();
        let host = dev.dtoh_sync_copy(&dst).unwrap();
        assert_eq!(&host, &truth);
    }

    #[test]
    fn test_pinned_copy_is_faster() {
        let dev = CudaDevice::new(0).unwrap();
        let n = 100_000;
        let n_samples = 5;
        let not_pinned = vec![0.0f32; n];

        let start = Instant::now();
        for _ in 0..n_samples {
            let _ = dev.htod_sync_copy(&not_pinned).unwrap();
            dev.synchronize().unwrap();
        }
        let unpinned_elapsed = start.elapsed() / n_samples;

        let pinned = unsafe { dev.alloc_pinned::<f32>(n) }.unwrap();

        let start = Instant::now();
        for _ in 0..n_samples {
            let _ = dev.htod_copy_pinned(&pinned).unwrap();
            dev.synchronize().unwrap();
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
