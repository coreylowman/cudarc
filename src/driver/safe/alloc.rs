use crate::driver::{result, sys};

use super::core::{CudaDevice, CudaSlice, CudaView, CudaViewMut};
use super::device_ptr::{DevicePtr, DevicePtrMut, DeviceSlice};

use std::{marker::Unpin, pin::Pin, sync::Arc, vec::Vec};

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

unsafe impl<T: DeviceRepr> DeviceRepr for &mut CudaSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<T: DeviceRepr> DeviceRepr for &CudaSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T: DeviceRepr> DeviceRepr for &CudaView<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T: DeviceRepr> DeviceRepr for &mut CudaViewMut<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

impl CudaDevice {
    /// Allocates device memory and increments the reference counter of [CudaDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    pub unsafe fn alloc_async<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let cu_device_ptr = result::malloc_async(self.stream, len * std::mem::size_of::<T>())?;
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
        })
    }

    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn alloc_zeros_async<T: ValidAsZeroBits + DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.alloc_async(len) }?;
        self.memset_zeros_async(&mut dst)?;
        Ok(dst)
    }

    /// Sets all memory to 0 asynchronously.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn memset_zeros_async<T: ValidAsZeroBits + DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        unsafe { result::memset_d8_async(*dst.device_ptr_mut(), 0, dst.num_bytes(), self.stream) }
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
    pub fn copy_dtod_async<T: DeviceRepr, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe {
            result::memcpy_dtod_async(
                *dst.device_ptr_mut(),
                *src.device_ptr(),
                src.len() * std::mem::size_of::<T>(),
                self.stream,
            )
        }
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_htod_async<T: Unpin + DeviceRepr>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.alloc_async(src.len()) }?;
        self.copy_into_htod_async(src, &mut dst)?;
        Ok(dst)
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_into_htod_async<T: DeviceRepr + Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        dst.host_buf = Some(Pin::new(src));
        unsafe {
            result::memcpy_htod_async(
                dst.cu_device_ptr,
                dst.host_buf.as_ref().unwrap(),
                self.stream,
            )
        }?;
        Ok(())
    }

    /// Allocates new device memory and synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::copy_htod_async()].
    ///
    /// # Safety
    ///
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_htod_sync<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &[T],
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.alloc_async(src.len()) }?;
        self.copy_into_htod_sync(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::copy_htod_async()].
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_into_htod_sync<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe { result::memcpy_htod_async(dst.cu_device_ptr, src, self.stream) }?;
        self.synchronize()
    }

    /// Synchronously copies device memory into host memory.
    /// Unlike [`CudaDevice::copy_into_dtoh_sync`] this returns a [`Vec<T>`].
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` (after returning) it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    #[allow(clippy::uninit_vec)]
    pub fn copy_dtoh_sync<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &CudaSlice<T>,
    ) -> Result<Vec<T>, result::DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        unsafe { dst.set_len(src.len()) };
        self.copy_into_dtoh_sync(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies device memory into host memory
    ///
    /// Use [`CudaDevice::copy_dtoh_sync`] if you need [`Vec<T>`] and can't provide
    /// a correctly sized slice.
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_into_dtoh_sync<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &CudaSlice<T>,
        dst: &mut [T],
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe { result::memcpy_dtoh_async(dst, src.cu_device_ptr, self.stream) }?;
        self.synchronize()
    }

    /// Synchronously de-allocates `src` and converts it into it's host value.
    /// You can just [drop] the slice if you don't need the host data.
    ///
    /// # Safety
    /// 1. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn reclaim<T: Clone + Default + DeviceRepr + Unpin>(
        self: &Arc<Self>,
        mut src: CudaSlice<T>,
    ) -> Result<Vec<T>, result::DriverError> {
        let buf = src.host_buf.take();
        let mut buf = buf.unwrap_or_else(|| {
            let mut b = Vec::with_capacity(src.len);
            b.resize(src.len, Default::default());
            Pin::new(b)
        });
        self.copy_into_dtoh_sync(&src, &mut buf)?;
        Ok(Pin::into_inner(buf))
    }

    /// Synchronizes the stream.
    pub fn synchronize(self: &Arc<Self>) -> Result<(), result::DriverError> {
        unsafe { result::stream::synchronize(self.stream) }
    }
}

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

#[cfg(test)]
mod tests {
    use crate::driver::CudaDeviceBuilder;

    use super::*;

    #[test]
    fn test_post_build_arc_count() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.alloc_zeros_async::<f32>(1).unwrap();
        assert!(t.host_buf.is_none());
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    fn test_post_take_arc_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.copy_htod_async([0.0f32; 5].to_vec()).unwrap();
        assert!(t.host_buf.is_some());
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.copy_htod_async([0.0f64; 10].to_vec()).unwrap();
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_arc_slice_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = Arc::new(device.copy_htod_async::<f64>([0.0; 10].to_vec()).unwrap());
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_release_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.copy_htod_async([1.0f32, 2.0, 3.0].to_vec()).unwrap();
        #[allow(clippy::redundant_clone)]
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);

        let r_host = device.reclaim(r).unwrap();
        assert_eq!(&r_host, &[1.0, 2.0, 3.0]);
        assert_eq!(Arc::strong_count(&device), 2);

        drop(r_host);
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let (free1, total1) = result::mem_get_info().unwrap();

        let t = device.copy_htod_async([0.0f32; 5].to_vec()).unwrap();
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
}
