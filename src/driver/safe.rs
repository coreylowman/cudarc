//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaDevice], [CudaDeviceBuilder], and more.
//!
//! # Usage
//!
//! 1. Instantiate a [CudaDevice] with [CudaDeviceBuilder]:
//!
//! ```rust
//! # use cudarc::driver::*;
//! let device = CudaDeviceBuilder::new(0).build().unwrap();
//! ```
//!
//! 2. Allocate device memory with host data with [CudaDevice::take_async()], [CudaDevice::alloc_zeros_async()],
//! or [CudaDevice::sync_copy()]
//!
//! You can also copy data to CudaSlice using [CudaDevice::sync_copy_into()]
//!
//! ```rust
//! # use cudarc::driver::*;
//! # let device = CudaDeviceBuilder::new(0).build().unwrap();
//! let a_dev: CudaSlice<f32> = device.alloc_zeros_async(10).unwrap();
//! let b_dev: CudaSlice<f32> = device.take_async(vec![0.0; 10]).unwrap();
//! let c_dev: CudaSlice<f32> = device.sync_copy(&[1.0, 2.0, 3.0]).unwrap();
//! ```
//!
//! 3. Transfer to host memory with [CudaDevice::sync_release()] or [CudaDevice::sync_copy_from()]
//!
//! ```rust
//! # use cudarc::driver::*;
//! # use std::rc::Rc;
//! # let device = CudaDeviceBuilder::new(0).build().unwrap();
//! let a_dev: CudaSlice<f32> = device.alloc_zeros_async(10).unwrap();
//! let mut a_buf: [f32; 10] = [1.0; 10];
//! device.sync_copy_from(&a_dev, &mut a_buf);
//! assert_eq!(a_buf, [0.0; 10]);
//! let a_host: Vec<f32> = device.sync_release(a_dev).unwrap();
//! assert_eq!(&a_host, &[0.0; 10]);
//! ```
//!
//! ## Mutating device memory - [CudaFunction]
//!
//! See [LaunchAsync] and [CudaFunction].
//!
//! In order to mutate device data, you need to use cuda kernels.
//!
//! Loading kernels is done with [CudaDeviceBuilder::with_ptx()]
//! and [CudaDeviceBuilder::with_ptx_from_file()]:
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! let device = CudaDeviceBuilder::new(0)
//!     .with_ptx(ptx, "module_name", &["my_function"])
//!     .build()
//!     .unwrap();
//! ```
//!
//! Retrieve the function using the registered module name & actual function name:
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDeviceBuilder::new(0).with_ptx(ptx, "module_name", &["my_function"]).build().unwrap();
//! let func: CudaFunction = device.get_func("module_name", "my_function").unwrap();
//! ```
//!
//! Asynchronously execute the kernel:
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDeviceBuilder::new(0).with_ptx(ptx, "module_key", &["my_function"]).build().unwrap();
//! # let func: CudaFunction = device.get_func("module_key", "my_function").unwrap();
//! let mut a = device.alloc_zeros_async::<f32>(10).unwrap();
//! let cfg = LaunchConfig::for_num_elems(10);
//! unsafe { func.launch_async(cfg, (&mut a,)) }.unwrap();
//! ```
//!
//! Note: Launching kernels is **extremely unsafe**. See [LaunchAsync] for more info.
//!
//! ## Sub slices of [CudaSlice]
//!
//! For some operations, it is necessary to only operate on a small part of a single [CudaSlice].
//! For example, the slice may represent a batch of items, and you want to run separate kernels
//! on each of the items in the batch.
//!
//! Use [CudaSlice::try_slice()] and [CudaSlice::try_slice_mut()] for this. The returned
//! views ([CudaView] and [CudaViewMut] hold references to the owning [CudaSlice],
//! so rust's ownership system handles safety here.
//!
//! These view structs can be used with [CudaFunction].
//!
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDeviceBuilder::new(0).with_ptx(ptx, "module_key", &["my_function"]).build().unwrap();
//! let mut a: CudaSlice<f32> = device.alloc_zeros_async::<f32>(3 * 10).unwrap();
//! for i_batch in 0..3 {
//!     let mut a_sub_view: CudaViewMut<f32> = a.try_slice_mut(i_batch * 10..).unwrap();
//!     let f: CudaFunction = device.get_func("module_key", "my_function").unwrap();
//!     let cfg = LaunchConfig::for_num_elems(10);
//!     unsafe { f.launch_async(cfg, (&mut a_sub_view,)) }.unwrap();
//! }
//! ```
//!
//! #### A note on implementation
//!
//! It would be possible to re-use [CudaSlice] itself for sub-slices, however that would involve adding
//! another structure underneath the hood that is wrapped in an [Arc] to minimize data cloning. Overall
//! it seemed more complex than the current implementation.
//!
//! # Safety
//!
//! There are a number of aspects to this, but at a high level this API utilizes [std::sync::Arc] to
//! control when [CudaDevice] can be dropped.
//!
//! ### Context/Stream lifetimes
//!
//! The first part of safety is ensuring that [sys::CUcontext], [sys::CUdevice], and [sys::CUstream] all
//! live the required amount of time (i.e. device outlives context, which outlives stream).
//!
//! This is accomplished by putting all of them inside one struct, the [CudaDevice]. There are other ways,
//! such as adding newtypes that carry lifetimes with them, but this approach was chosen to make working
//! with device pointers easier.
//!
//! Additionally, [CudaDevice] implements [Drop] as releasing all the data from the device in
//! the expected way.
//!
//! ### Device Data lifetimes
//!
//! The next part of safety is ensuring that [CudaSlice] do not outlive
//! the [CudaDevice]. For usability, each [CudaSlice] owns an [`Arc<CudaDevice>`]
//! to ensure the device stays alive.
//!
//! Additionally we don't want to double free any device pointers, so free is only
//! called when the device pointer is dropped. Thanks rust!
//!
//! ### Host and Device Data lifetimes
//!
//! Each device allocation can be associated with a host allocation. We want to ensure
//! that these have the same lifetimes *when copying data between them*.
//!
//! This is done via the various copy methods. Methods that don't take ownership
//! of the host data need to be executed synchronously, while the methods own the reference.
//! Methods that do own the host data can be executed synchronously.
//!
//! ### Single stream operations
//!
//! The next part of safety is ensuring that all operations happen on a single stream.
//! This ensures that data isn't mutated by more than 1 stream at a time, and also
//! ensures data isn't used before allocated, or used after free.
//!
//! At the moment, only a single stream is supported, and only the `*_async` methods
//! in [crate::driver::result] are used.
//!
//! Another important aspect of this is ensuring that mutability in an async setting
//! is sound, and something can't be freed while it's being used in a kernel.
//!
//! Unfortunately, it also is inefficient to keep all `free()` operations on the
//! same stream as actual work.
//!
//! To this end [CudaDevice] actual has a 2nd stream, where it places all `free()`
//! operations. These are synchronized with the main stream using the [result::event]
//! module and [result::stream::wait_event].

use super::{result, sys};
use crate::nvrtc::Ptx;

use alloc::ffi::{CString, NulError};
use core::ops::{Bound, RangeBounds};
use spin::RwLock;
use std::{collections::BTreeMap, marker::Unpin, pin::Pin, sync::Arc, vec::Vec};

pub use result::DriverError;

/// Contains a reference counted pointer to both
/// device and host memory allocated for type `T`.
///
/// # Host data
///
/// *This owns the host data it is associated with*. However
/// it is possible to create device memory without having
/// a corresponding host memory, so the host memory is
/// actually [Option].
///
/// # Reclaiming host data
///
/// To reclaim the host data for this device data,
/// use [CudaDevice::sync_release()]. This will
/// perform necessary synchronization to ensure
/// that the device data finishes copying over.
///
/// # Mutating device data
///
/// This can only be done by launching kernels via
/// [LaunchAsync] which is implemented
/// by [CudaDevice]. Pass `&mut CudaSlice<T>`
/// if you want to mutate the rc, and `&CudaSlice<T>` otherwise.
///
/// Unfortunately, `&CudaSlice<T>` can **still be mutated
/// by the [CudaFunction]**.
#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) host_buf: Option<Pin<Vec<T>>>,
}

unsafe impl<T: Send> Send for CudaSlice<T> {}
unsafe impl<T: Sync> Sync for CudaSlice<T> {}

impl<T> CudaSlice<T> {
    /// Number of elements in the slice
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of the slice in bytes
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn clone_async(&self) -> Result<Self, DriverError> {
        let dst = unsafe { self.device.alloc_async(self.len) }?;
        unsafe {
            result::memcpy_dtod_async(
                dst.cu_device_ptr,
                self.cu_device_ptr,
                self.num_bytes(),
                self.device.stream,
            )
        }?;
        Ok(dst)
    }
}

impl<T> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.clone_async().unwrap()
    }
}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        unsafe {
            // 1. record the current state of cu_stream on the event.
            result::event::record(self.device.event, self.device.stream).unwrap();

            // 2. make dealloc_stream wait for the event to be marked as complete.
            result::stream::wait_event(
                self.device.free_stream,
                self.device.event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
            .unwrap();

            // 3. add a free operation to the dealloc stream.
            // Since we just made dealloc_stream wait on the event, which is synchronized
            // on the device stream, that means this free will not occur until all currently
            // existing jobs on the stream execute.
            result::free_async(self.cu_device_ptr, self.device.free_stream).unwrap();
        }
    }
}

impl<T: Clone + Default + Unpin> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.device.clone().sync_release(value)
    }
}

trait RangeHelper<T: PartialOrd> {
    fn inclusive_start(&self, valid: &impl RangeBounds<T>) -> Option<T>;
    fn inclusive_end(&self, valid: &impl RangeBounds<T>) -> Option<T>;
    fn bounds(&self, valid: impl RangeBounds<T>) -> Option<(T, T)> {
        self.inclusive_start(&valid).and_then(|s| {
            self.inclusive_end(&valid)
                .and_then(|e| (s <= e).then_some((s, e)))
        })
    }
}
impl<R: RangeBounds<usize>> RangeHelper<usize> for R {
    fn inclusive_start(&self, valid: &impl RangeBounds<usize>) -> Option<usize> {
        match self.start_bound() {
            Bound::Included(n) if valid.contains(n) => Some(*n),
            Bound::Excluded(n) if n < &usize::MAX && valid.contains(&(*n + 1)) => Some(*n + 1),
            Bound::Unbounded => valid.inclusive_start(&(0..=usize::MAX)),
            _ => None,
        }
    }
    fn inclusive_end(&self, valid: &impl RangeBounds<usize>) -> Option<usize> {
        match self.end_bound() {
            Bound::Included(n) if valid.contains(n) => Some(*n),
            Bound::Excluded(n) if n > &0 && valid.contains(&(*n - 1)) => Some(*n - 1),
            Bound::Unbounded => valid.inclusive_end(&(0..=usize::MAX)),
            _ => None,
        }
    }
}

/// A immutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice()],
/// which implements [AsKernelParam] for use with kernels.
///
/// See module docstring for more details.
#[allow(unused)]
pub struct CudaView<'a, T> {
    slice: &'a CudaSlice<T>,
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl<'a, T> CudaView<'a, T> {
    /// Number of elements in the slice
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of the slice in bytes
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// A mutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice_mut()],
/// which implements [AsKernelParam] for use with kernels.
///
/// See module docstring for more details.
#[allow(unused)]
pub struct CudaViewMut<'a, T> {
    slice: &'a mut CudaSlice<T>,
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl<'a, T> CudaViewMut<'a, T> {
    /// Number of elements in the slice
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of the slice in bytes
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T> CudaSlice<T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `range.start >= self.len`
    ///
    /// See module docstring for example
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaView {
            ptr: self.cu_device_ptr + (start * std::mem::size_of::<T>()) as u64,
            slice: self,
            len: 1 + end - start,
        })
    }

    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `offset >= self.len`
    ///
    /// See module docstring for example
    pub fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaViewMut {
            ptr: self.cu_device_ptr + (start * std::mem::size_of::<T>()) as u64,
            slice: self,
            len: 1 + end - start,
        })
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr;
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.cu_device_ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> DevicePtr<T> for CudaView<'a, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr;
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.cu_device_ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> DevicePtrMut<T> for CudaViewMut<'a, T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// A wrapper around [sys::CUdevice], [sys::CUcontext], [sys::CUstream],
/// and [CudaFunction].
///
/// **Must be created through [CudaDeviceBuilder].**
///
/// # Safety
/// 1. impl [Drop] to call all the corresponding resource cleanup methods
/// 2. Doesn't impl clone, so you can't have multiple device pointers
/// hanging around.
/// 3. Any allocations enforce that self is an [Arc], meaning no allocation
/// can outlive the [CudaDevice]
#[derive(Debug)]
pub struct CudaDevice {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_primary_ctx: sys::CUcontext,
    /// The stream that all work is executed on.
    pub(crate) stream: sys::CUstream,
    /// A stream that only contains free_async calls so they don't block the `stream`.
    pub(crate) free_stream: sys::CUstream,
    /// Used to synchronize `free_stream` & `stream`
    pub(crate) event: sys::CUevent,
    pub(crate) modules: RwLock<BTreeMap<&'static str, CudaModule>>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        let modules = RwLock::get_mut(&mut self.modules);
        for (_, module) in modules.iter() {
            unsafe { result::module::unload(module.cu_module) }.unwrap();
        }
        modules.clear();

        let stream = std::mem::replace(&mut self.stream, std::ptr::null_mut());
        if !stream.is_null() {
            unsafe { result::stream::destroy(stream) }.unwrap();
        }

        let stream = std::mem::replace(&mut self.free_stream, std::ptr::null_mut());
        if !stream.is_null() {
            unsafe { result::stream::destroy(stream) }.unwrap();
        }

        let event = std::mem::replace(&mut self.event, std::ptr::null_mut());
        if !event.is_null() {
            unsafe { result::event::destroy(event) }.unwrap();
        }

        let ctx = std::mem::replace(&mut self.cu_primary_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            unsafe { result::primary_ctx::release(self.cu_device) }.unwrap();
        }
    }
}

impl CudaDevice {
    /// Allocates device memory and increments the reference counter of [CudaDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    pub unsafe fn alloc_async<T>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
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
    pub fn alloc_zeros_async<T: ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc_async(len) }?;
        self.memset_zeros_async(&mut dst)?;
        Ok(dst)
    }

    /// Sets all memory to 0 asynchronously.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn memset_zeros_async<T: ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        unsafe { result::memset_d8_async(*dst.device_ptr_mut(), 0, dst.num_bytes(), self.stream) }
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn take_async<T: Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc_async(src.len()) }?;
        self.copy_into_async(src, &mut dst)?;
        Ok(dst)
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
    pub fn device_copy_async<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
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

    /// Allocates new device memory and synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::take_async()].
    ///
    /// # Safety
    ///
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy<T>(self: &Arc<Self>, src: &[T]) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc_async(src.len()) }?;
        self.sync_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::take_async()].
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy_into<T>(
        self: &Arc<Self>,
        src: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe { result::memcpy_htod_async(dst.cu_device_ptr, src, self.stream) }?;
        self.synchronize()
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn copy_into_async<T: Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), DriverError> {
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

    /// Synchronously copies device memory into host memory
    ///
    /// Use [`CudaDevice::sync_copy_into_vec`] if you need [`Vec<T>`] and can't provide
    /// a correctly sized slice.
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy_from<T>(
        self: &Arc<Self>,
        src: &CudaSlice<T>,
        dst: &mut [T],
    ) -> Result<(), DriverError> {
        assert_eq!(src.len(), dst.len());
        unsafe { result::memcpy_dtoh_async(dst, src.cu_device_ptr, self.stream) }?;
        self.synchronize()
    }

    /// Synchronously copies device memory into host memory.
    /// Unlike [`CudaDevice::sync_copy_from`] this returns a [`Vec<T>`].
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` (after returning) it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_copy_into_vec<T>(
        self: &Arc<Self>,
        src: &CudaSlice<T>,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        unsafe {
            dst.set_len(src.len());
            result::memcpy_dtoh_async(dst.as_mut_slice(), src.cu_device_ptr, self.stream)
        }?;
        self.synchronize()?;
        Ok(dst)
    }

    /// De-allocates `src` and converts it into it's host value. You can just drop the slice if you don't
    /// need the host data.
    ///
    /// # Safety
    /// 1. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_release<T: Clone + Default + Unpin>(
        self: &Arc<Self>,
        mut src: CudaSlice<T>,
    ) -> Result<Vec<T>, DriverError> {
        let buf = src.host_buf.take();
        let mut buf = buf.unwrap_or_else(|| {
            let mut b = Vec::with_capacity(src.len);
            b.resize(src.len, Default::default());
            Pin::new(b)
        });
        self.sync_copy_from(&src, &mut buf)?;
        Ok(Pin::into_inner(buf))
    }

    /// Synchronizes the stream.
    pub fn synchronize(self: &Arc<Self>) -> Result<(), DriverError> {
        unsafe { result::stream::synchronize(self.stream) }
    }

    /// Whether a module and function are currently loaded into the device.
    pub fn has_func(self: &Arc<Self>, module_name: &str, func_name: &str) -> bool {
        let modules = self.modules.read();
        modules
            .get(module_name)
            .map_or(false, |module| module.has_func(func_name))
    }

    /// Retrieves a [CudaFunction] that was registered under `module_name` and `func_name`.
    pub fn get_func(self: &Arc<Self>, module_name: &str, func_name: &str) -> Option<CudaFunction> {
        let modules = self.modules.read();
        modules
            .get(module_name)
            .and_then(|m| m.get_func(func_name))
            .map(|cu_function| CudaFunction {
                cu_function,
                device: self.clone(),
            })
    }

    /// Dynamically load a set of [CudaFunction] from a ptx file. See [CudaDeviceBuilder::with_ptx_from_file].
    pub fn load_ptx_from_file(
        self: &Arc<Self>,
        ptx_path: &'static str,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), BuildError> {
        let m = CudaDeviceBuilder::build_module_from_ptx_file(ptx_path, module_name, func_names)?;
        {
            let mut modules = self.modules.write();
            modules.insert(module_name, m);
        }
        Ok(())
    }

    /// Dynamically load a set of [CudaFunction] from a jit compiled ptx. See [CudaDeviceBuilder::with_ptx]
    pub fn load_ptx(
        self: &Arc<Self>,
        ptx: Ptx,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), BuildError> {
        let m = CudaDeviceBuilder::build_module_from_ptx(ptx, module_name, func_names)?;
        {
            let mut modules = self.modules.write();
            modules.insert(module_name, m);
        }
        Ok(())
    }
}

/// A wrapper around [sys::CUstream] that safely ensures null stream is synchronized
/// upon the completion of this streams work.
///
/// Create with [CudaDevice::auto_joining_stream].
///
/// The synchronization happens in **code order**. E.g.
/// ```ignore
/// let stream = dev.auto_joining_stream()?; // 0
/// dev.launch_async(...)?; // 1
/// dev.par_launch_async(&stream, ...)?; // 2
/// dev.launch_async(...)?; // 3
/// drop(stream); // 4
/// dev.launch_async(...) // 5
/// ```
///
/// - 0 will place a streamWaitEvent(default work stream) on the new stream
/// - 1 will launch on the default work stream
/// - 2 will launch concurrently to 1 on `&stream`,
/// - 3 will launch after 1 on the default work stream, but potentially concurrently to 2.
/// - 4 will place a streamWaitEvent(`&stream`) on default work stream
/// - 5 will happen on the default stream **after the default stream waits for 2**
#[derive(Debug)]
pub struct CudaStream {
    pub stream: sys::CUstream,
    device: Arc<CudaDevice>,
}

impl CudaDevice {
    /// Allocates a new stream that can execute kernels concurrently to the default stream.
    ///
    /// This stream synchronizes in the following way:
    /// 1. On creation it adds a wait for any existing work on the default work stream to complete
    /// 2. On drop it adds a wait for any existign work on Self to complete *to the default stream*.
    pub fn auto_joining_stream(self: &Arc<Self>) -> Result<CudaStream, DriverError> {
        let stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;
        unsafe {
            result::event::record(self.event, self.stream)?;
            result::stream::wait_event(
                stream,
                self.event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )?;
        }
        Ok(CudaStream {
            stream,
            device: self.clone(),
        })
    }

    /// Forces [CudaStream] to drop, causing the default work stream to block on `streams` completion.
    #[allow(unused_variables)]
    pub fn join_async(self: &Arc<Self>, stream: CudaStream) -> Result<(), DriverError> {
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            result::event::record(self.device.event, self.stream).unwrap();
            result::stream::wait_event(
                self.device.stream,
                self.device.event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
            .unwrap();
            result::stream::destroy(self.stream).unwrap();
        }
    }
}

/// Wrapper around [sys::CUmodule] that also contains
/// the loaded [CudaFunction] associated with this module.
///
/// See [CudaModule::get_fn()] for retrieving function handles.
///
/// See [CudaDeviceBuilder] for how to construct these modules.
#[derive(Debug)]
pub(crate) struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: BTreeMap<&'static str, sys::CUfunction>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Returns reference to function with `name`. If function
    /// was not already loaded into CudaModule, then `None`
    /// is returned.
    pub(crate) fn get_func(&self, name: &str) -> Option<sys::CUfunction> {
        self.functions.get(name).cloned()
    }

    pub(crate) fn has_func(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

/// Wrapper around [sys::CUfunction]. Used by [LaunchAsync].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    pub(crate) device: Arc<CudaDevice>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {
    #[inline]
    unsafe fn launch_async_impl(
        self,
        cfg: LaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.device.stream,
            params,
        )
    }

    #[inline]
    unsafe fn par_launch_async_impl(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.stream,
            params,
        )
    }
}

/// Configuration for [result::launch_kernel]
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
/// for description of each parameter.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a [LaunchConfig] with:
    /// - block_dim == `1024`
    /// - grid_dim == `(n - 1023) / 1024`
    /// - shared_mem_bytes == `0`
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        Self {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Something that can be turned into a parameter for
/// [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because it can take any type and
/// turn it into a mutable pointer.
///
/// Additionally, all the safety notices for [result::launch_kernel]
/// apply here as well.
pub unsafe trait AsKernelParam {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

unsafe impl AsKernelParam for i8 {}
unsafe impl AsKernelParam for i16 {}
unsafe impl AsKernelParam for i32 {}
unsafe impl AsKernelParam for i64 {}
unsafe impl AsKernelParam for i128 {}
unsafe impl AsKernelParam for isize {}
unsafe impl AsKernelParam for u8 {}
unsafe impl AsKernelParam for u16 {}
unsafe impl AsKernelParam for u32 {}
unsafe impl AsKernelParam for u64 {}
unsafe impl AsKernelParam for u128 {}
unsafe impl AsKernelParam for usize {}
unsafe impl AsKernelParam for f32 {}
unsafe impl AsKernelParam for f64 {}
#[cfg(feature = "f16")]
unsafe impl AsKernelParam for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl AsKernelParam for half::bf16 {}

unsafe impl<T> AsKernelParam for &mut CudaSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<T> AsKernelParam for &CudaSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T> AsKernelParam for &CudaView<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T> AsKernelParam for &mut CudaViewMut<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

/// Consumes a [CudaFunction] to execute asychronously on the device with
/// params determined by generic parameter `Params`.
///
/// This is impl'd multiple times for different number and types of params. In
/// general, `Params` should impl [AsKernelParam]
///
/// # Safety
///
/// This is not safe really ever, because there's no garuntee that `Params`
/// will work for any [CudaFunction] passed in. Great care should be taken
/// to ensure that [CudaFunction] works with `Params` and that the correct
/// parameters have `&mut` in front of them.
///
/// Additionally, kernels can mutate data that is marked as immutable,
/// such as `&CudaSlice<T>`.
///
/// See [LaunchAsync::launch_async] for more details
pub unsafe trait LaunchAsync<Params> {
    /// Launches the [CudaFunction] with the corresponding `Params`.
    ///
    /// # Safety
    ///
    /// This method is **very** unsafe.
    ///
    /// See cuda documentation notes on this as well:
    /// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#functions>
    ///
    /// 1. `params` can be changed regardless of `&` or `&mut` usage.
    /// 2. `params` will be changed at some later point after the
    /// function returns because the kernel is executed async.
    /// 3. There are no guaruntees that the `params`
    /// are the correct number/types/order for `func`.
    /// 4. Specifying the wrong values for [LaunchConfig] can result
    /// in accessing/modifying values past memory limits.
    ///
    /// ## Asynchronous mutation
    ///
    /// Since this library queues kernels to be launched on a single
    /// stream, and really the only way to modify [CudaSlice] is through
    /// kernels, mutating the same [CudaSlice] with multiple kernels
    /// is safe. This is because each kernel is executed sequentially
    /// on the stream.
    ///
    /// **Modifying a value on the host that is in used by a
    /// kernel is undefined behavior.** But is hard to do
    /// accidentally.
    ///
    /// Also for this reason, do not pass in any values to kernels
    /// that can be modified on the host. This is the reason
    /// [AsKernelParam] is not implemented for rust primitive
    /// references.
    ///
    /// ## Use after free
    ///
    /// Since the drop implementation for [CudaSlice] also occurs
    /// on the device's single stream, any kernels launched before
    /// the drop will complete before the value is actually freed.
    ///
    /// **If you launch a kernel or drop a value on a different stream
    /// this may not hold**
    unsafe fn launch_async(self, cfg: LaunchConfig, params: Params) -> Result<(), DriverError>;

    /// Launch the function on a stream concurrent to the device's default
    /// work stream.
    ///
    /// # Safety
    /// This method is even more unsafe than [LaunchAsync::launch_async], all the same rules apply,
    /// except now things are executing in parallel to each other.
    ///
    /// That means that if any of the kernels modify the same memory location, you'll get race
    /// conditions or potentially undefined behavior.
    unsafe fn par_launch_async(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), DriverError>;
}

macro_rules! impl_launch {
    ([$($Vars:tt),*], [$($Idx:tt),*]) => {
unsafe impl<$($Vars: AsKernelParam),*> LaunchAsync<($($Vars, )*)> for CudaFunction {
    unsafe fn launch_async(
        self,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), DriverError> {
        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
        self.launch_async_impl(cfg, params)
    }

    unsafe fn par_launch_async(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), DriverError> {
        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
        self.par_launch_async_impl(stream, cfg, params)
    }
}
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);
impl_launch!([A, B, C, D, E, F, G, H, I], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);

/// A builder for [CudaDevice].
///
/// Call [CudaDeviceBuilder::new()] to start, and [CudaDeviceBuilder::build]
/// to finish.
///
/// Provides a way to specify what modules & functions to load into
/// the device via [CudaDeviceBuilder::with_ptx_from_file()]
/// and [CudaDeviceBuilder::with_ptx()].
#[derive(Debug)]
pub struct CudaDeviceBuilder {
    pub(crate) ordinal: usize,
    pub(crate) ptx_files: Vec<PtxFileConfig>,
    pub(crate) ptxs: Vec<PtxConfig>,
}

#[derive(Debug)]
pub(crate) struct PtxFileConfig {
    pub(crate) key: &'static str,
    pub(crate) fname: &'static str,
    pub(crate) fn_names: Vec<&'static str>,
}

#[derive(Debug)]
pub(crate) struct PtxConfig {
    pub(crate) key: &'static str,
    pub(crate) ptx: Ptx,
    pub(crate) fn_names: Vec<&'static str>,
}

impl CudaDeviceBuilder {
    /// Starts a new builder object.
    /// - `ordinal` is the index of th cuda device to attach to.
    pub fn new(ordinal: usize) -> Self {
        Self {
            ordinal,
            ptx_files: Vec::new(),
            ptxs: Vec::new(),
        }
    }

    /// Adds a path to a precompiled `.ptx` file to be loaded as a module on the device.
    ///
    /// - `ptx_path` is a file
    /// - `key` is a unique identifier used to access the module later on with [CudaDevice::get_func()]
    /// - `fn_names` is a slice of function names to load into the module during build.
    pub fn with_ptx_from_file(
        mut self,
        ptx_path: &'static str,
        key: &'static str,
        fn_names: &[&'static str],
    ) -> Self {
        self.ptx_files.push(PtxFileConfig {
            key,
            fname: ptx_path,
            fn_names: fn_names.to_vec(),
        });
        self
    }

    /// Add a [Ptx] compiled with nvrtc to be loaded as a module on the device.
    ///
    /// - `key` is a unique identifier used to access the module later on with [CudaDevice::get_func()]
    /// - `ptx` contains the compilex ptx
    /// - `fn_names` is a slice of function names to load into the module during build.
    pub fn with_ptx(mut self, ptx: Ptx, key: &'static str, fn_names: &[&'static str]) -> Self {
        self.ptxs.push(PtxConfig {
            key,
            ptx,
            fn_names: fn_names.to_vec(),
        });
        self
    }

    /// Builds the [CudaDevice]:
    /// 1. Initializes cuda with [result::init]
    /// 2. Creates the device/primary ctx, and stream
    /// 3. Uses nvrtc to compile and the modules & functions
    pub fn build(mut self) -> Result<Arc<CudaDevice>, BuildError> {
        result::init().map_err(BuildError::InitError)?;

        let cu_device =
            result::device::get(self.ordinal as i32).map_err(BuildError::DeviceError)?;

        // primary context initialization
        let cu_primary_ctx =
            unsafe { result::primary_ctx::retain(cu_device) }.map_err(BuildError::ContextError)?;

        unsafe { result::ctx::set_current(cu_primary_ctx) }.map_err(BuildError::ContextError)?;

        let free_stream = result::stream::create(result::stream::StreamKind::NonBlocking)
            .map_err(BuildError::StreamError)?;

        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .map_err(BuildError::DeviceError)?;

        let mut modules = BTreeMap::new();

        for cu in self.ptx_files.drain(..) {
            modules.insert(
                cu.key,
                Self::build_module_from_ptx_file(cu.fname, cu.key, &cu.fn_names)?,
            );
        }

        for ptx in self.ptxs.drain(..) {
            modules.insert(
                ptx.key,
                Self::build_module_from_ptx(ptx.ptx, ptx.key, &ptx.fn_names)?,
            );
        }

        let device = CudaDevice {
            cu_device,
            cu_primary_ctx,
            stream: std::ptr::null_mut(),
            free_stream,
            event,
            modules: RwLock::new(modules),
        };
        Ok(Arc::new(device))
    }

    fn build_module_from_ptx_file(
        ptx_path: &'static str,
        key: &'static str,
        func_names: &[&'static str],
    ) -> Result<CudaModule, BuildError> {
        let name_c = CString::new(ptx_path).map_err(BuildError::CStringError)?;
        let cu_module = result::module::load(name_c)
            .map_err(|cuda| BuildError::PtxLoadingError { key, cuda })?;
        let mut functions = BTreeMap::new();
        for &fn_name in func_names.iter() {
            let fn_name_c = CString::new(fn_name).map_err(BuildError::CStringError)?;
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }
                .map_err(|e| BuildError::GetFunctionError {
                    key,
                    symbol: fn_name,
                    cuda: e,
                })?;
            functions.insert(fn_name, cu_function);
        }
        Ok(CudaModule {
            cu_module,
            functions,
        })
    }

    fn build_module_from_ptx(
        ptx: Ptx,
        key: &'static str,
        fn_names: &[&'static str],
    ) -> Result<CudaModule, BuildError> {
        let cu_module = match ptx {
            Ptx::Image(image) => unsafe { result::module::load_data(image.as_ptr() as *const _) },
            Ptx::Src(src) => {
                let c_src = CString::new(src).unwrap();
                unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
            }
        }
        .map_err(|cuda| BuildError::NvrtcLoadingError { key, cuda })?;
        let mut functions = BTreeMap::new();
        for &fn_name in fn_names.iter() {
            let fn_name_c = CString::new(fn_name).map_err(BuildError::CStringError)?;
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }
                .map_err(|e| BuildError::GetFunctionError {
                    key,
                    symbol: fn_name,
                    cuda: e,
                })?;
            functions.insert(fn_name, cu_function);
        }
        Ok(CudaModule {
            cu_module,
            functions,
        })
    }
}

/// Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then profiler_start() has no effect.
/// ```rust
/// use cudarc::driver::{profiler_start, profiler_stop};
///
/// profiler_start()?;
/// // Hotpath
/// profiler_stop()?;
/// // Now check your results
/// // nsys profile -c cudaProfilerApi /path/to/bin
/// // And this will profile only the hotpath.
/// ```
///
pub fn profiler_start() -> Result<(), DriverError> {
    unsafe { sys::cuProfilerStart() }.result()
}

/// Disables profile collection by the active profiling tool for the current context. If profiling is already disabled, then profiler_stop() has no effect.
pub fn profiler_stop() -> Result<(), DriverError> {
    unsafe { sys::cuProfilerStop() }.result()
}

/// An error the occurs during [CudaDeviceBuilder::build]
#[derive(Debug)]
pub enum BuildError {
    InitError(DriverError),
    DeviceError(DriverError),
    ContextError(DriverError),
    StreamError(DriverError),
    PtxLoadingError {
        key: &'static str,
        cuda: DriverError,
    },
    NvrtcLoadingError {
        key: &'static str,
        cuda: DriverError,
    },
    GetFunctionError {
        key: &'static str,
        symbol: &'static str,
        cuda: DriverError,
    },
    CStringError(NulError),
}

#[cfg(feature = "std")]
impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {}

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
    use std::time::Instant;

    use crate::nvrtc::compile_ptx_with_opts;

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
        let t = device.take_async([0.0f32; 5].to_vec()).unwrap();
        assert!(t.host_buf.is_some());
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_counts() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async([0.0f64; 10].to_vec()).unwrap();
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
        let t = Arc::new(device.take_async([0.0f64; 10].to_vec()).unwrap());
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
        let t = device.take_async([1.0f32, 2.0, 3.0].to_vec()).unwrap();
        #[allow(clippy::redundant_clone)]
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);

        let r_host = device.sync_release(r).unwrap();
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

        let t = device.take_async([0.0f32; 5].to_vec()).unwrap();
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
    fn test_mut_into_kernel_param_no_inc_rc() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async([0.0f32; 1].to_vec()).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).as_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    #[test]
    fn test_ref_into_kernel_param_inc_rc() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let t = device.take_async([0.0f32; 1].to_vec()).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).as_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    const SIN_CU: &str = "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}";

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "sin", &["sin_kernel"])
            .build()
            .unwrap();
        let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];

        let a_dev = dev.take_async(a_host.clone().to_vec()).unwrap();

        let mut b_dev = a_dev.clone();

        unsafe {
            sin_kernel.launch_async(
                LaunchConfig::for_num_elems(10),
                (&mut b_dev, &a_dev, 10usize),
            )
        }
        .unwrap();

        let b_host = dev.sync_release(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    #[test]
    fn test_large_launches() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "sin", &["sin_kernel"])
            .build()
            .unwrap();
        for numel in [256, 512, 1024, 1280, 1536, 2048] {
            let mut a = Vec::with_capacity(numel);
            a.resize(numel, 1.0f32);

            let a = dev.take_async(a).unwrap();
            let mut b = dev.alloc_zeros_async::<f32>(numel).unwrap();

            let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            unsafe { sin_kernel.launch_async(cfg, (&mut b, &a, numel)) }.unwrap();

            let b = dev.sync_release(b).unwrap();
            for v in b {
                assert_eq!(v, 0.841471);
            }
        }
    }

    #[test]
    fn test_bounds_helper() {
        assert_eq!((..2usize).bounds(0..=usize::MAX), Some((0, 1)));
        assert_eq!((1..2usize).bounds(..=usize::MAX), Some((1, 1)));
        assert_eq!((..).bounds(1..10), Some((1, 9)));
        assert_eq!((2..=2usize).bounds(0..=usize::MAX), Some((2, 2)));
        assert_eq!((2..=2usize).bounds(0..=1), None);
        assert_eq!((2..2usize).bounds(0..=usize::MAX), None);
    }

    #[test]
    fn test_device_copy_to_views() {
        let dev = CudaDeviceBuilder::new(0).build().unwrap();

        let smalls = [
            dev.take_async(std::vec![-1.0f32, -0.8]).unwrap(),
            dev.take_async(std::vec![-0.6, -0.4]).unwrap(),
            dev.take_async(std::vec![-0.2, 0.0]).unwrap(),
            dev.take_async(std::vec![0.2, 0.4]).unwrap(),
            dev.take_async(std::vec![0.6, 0.8]).unwrap(),
        ];
        let mut big = dev.alloc_zeros_async::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.try_slice_mut(offset..offset + small.len()).unwrap();
            dev.device_copy_async(small, &mut sub).unwrap();
            offset += small.len();
        }

        assert_eq!(
            dev.sync_release(big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }

    #[test]
    fn test_launch_with_views() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "sin", &["sin_kernel"])
            .build()
            .unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
        let a_dev = dev.take_async(a_host.clone().to_vec()).unwrap();
        let mut b_dev = a_dev.clone();

        for i in 0..5 {
            let a_sub = a_dev.try_slice(i * 2..).unwrap();
            assert_eq!(a_sub.len, 10 - 2 * i);
            let mut b_sub = b_dev.try_slice_mut(i * 2..).unwrap();
            assert_eq!(b_sub.len, 10 - 2 * i);
            let f = dev.get_func("sin", "sin_kernel").unwrap();
            unsafe { f.launch_async(LaunchConfig::for_num_elems(2), (&mut b_sub, &a_sub, 2usize)) }
                .unwrap();
        }

        let b_host = dev.sync_release(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    const TEST_KERNELS: &str = "
extern \"C\" __global__ void int_8bit(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    assert(s_min == -128);
    assert(s_max == 127);
    assert(u_min == 0);
    assert(u_max == 255);
}

extern \"C\" __global__ void int_16bit(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    assert(s_min == -32768);
    assert(s_max == 32767);
    assert(u_min == 0);
    assert(u_max == 65535);
}

extern \"C\" __global__ void int_32bit(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    assert(s_min == -2147483648);
    assert(s_max == 2147483647);
    assert(u_min == 0);
    assert(u_max == 4294967295);
}

extern \"C\" __global__ void int_64bit(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    assert(s_min == -9223372036854775808);
    assert(s_max == 9223372036854775807);
    assert(u_min == 0);
    assert(u_max == 18446744073709551615);
}

extern \"C\" __global__ void floating(float f, double d) {
    assert(fabs(f - 1.2345678) <= 1e-7);
    assert(fabs(d - -10.123456789876543) <= 1e-16);
}
";

    #[test]
    fn test_launch_with_8bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_8bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_8bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i8::MIN, i8::MAX, u8::MIN, u8::MAX),
            )
        }
        .unwrap();

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_16bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_16bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_16bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i16::MIN, i16::MAX, u16::MIN, u16::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_32bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_32bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_32bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i32::MIN, i32::MAX, u32::MIN, u32::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_64bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["int_64bit"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "int_64bit").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (i64::MIN, i64::MAX, u64::MIN, u64::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_floats() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["floating"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "floating").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (1.2345678f32, -10.123456789876543f64),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[cfg(feature = "f16")]
    const HALF_KERNELS: &str = "
#include \"cuda_fp16.h\"

extern \"C\" __global__ void halfs(__half h) {
    assert(__habs(h - __float2half(1.234)) <= __float2half(1e-4));
}
";

    #[cfg(feature = "f16")]
    #[test]
    fn test_launch_with_half() {
        use crate::nvrtc::CompileOptions;

        let ptx = compile_ptx_with_opts(
            HALF_KERNELS,
            CompileOptions {
                include_paths: std::vec!["/usr/include".into()],
                arch: Some("compute_53"),
                ..Default::default()
            },
        )
        .unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["halfs"])
            .build()
            .unwrap();
        let f = dev.get_func("tests", "halfs").unwrap();
        unsafe {
            f.launch_async(
                LaunchConfig::for_num_elems(1),
                (half::f16::from_f32(1.234),),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    const SLOW_KERNELS: &str = "
extern \"C\" __global__ void slow_worker(const float *data, const size_t len, float *out) {
    float tmp = 0.0;
    for(size_t i = 0; i < 1000000; i++) {
        tmp += data[i % len];
    }
    *out = tmp;
}
";

    #[test]
    fn test_par_launch() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let dev = CudaDeviceBuilder::new(0)
            .with_ptx(ptx, "tests", &["slow_worker"])
            .build()
            .unwrap();
        let slice = dev.alloc_zeros_async::<f32>(1000)?;
        let mut a = dev.alloc_zeros_async::<f32>(1)?;
        let mut b = dev.alloc_zeros_async::<f32>(1)?;
        let cfg = LaunchConfig::for_num_elems(1);

        let start = Instant::now();
        {
            // launch two kernels on the default stream
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch_async(cfg, (&slice, slice.len(), &mut a))? };
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch_async(cfg, (&slice, slice.len(), &mut b))? };
            dev.synchronize()?;
        }
        let double_launch_s = start.elapsed().as_secs_f64();

        let start = Instant::now();
        {
            // create a new stream & launch them concurrently
            let stream = dev.auto_joining_stream()?;
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch_async(cfg, (&slice, slice.len(), &mut a))? };
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.par_launch_async(&stream, cfg, (&slice, slice.len(), &mut b))? };
            dev.synchronize()?;
        }
        let par_launch_s = start.elapsed().as_secs_f64();

        assert!(
            (double_launch_s - 2.0 * par_launch_s).abs() < 20.0 / 1000.0,
            "par={:?} dbl={:?}",
            par_launch_s,
            double_launch_s
        );
        Ok(())
    }
}
