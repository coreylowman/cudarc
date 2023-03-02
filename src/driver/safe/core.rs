use crate::driver::{result, sys};

use super::{alloc::DeviceRepr, device_ptr::DeviceSlice};

use core::ops::{Bound, RangeBounds};
use spin::RwLock;
use std::{collections::BTreeMap, marker::Unpin, pin::Pin, sync::Arc, vec::Vec};

/// A wrapper around [sys::CUdevice], [sys::CUcontext], [sys::CUstream],
/// and [CudaFunction].
///
/// **Must be created through [crate::driver::CudaDeviceBuilder].**
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

unsafe impl DeviceRepr for CudaDevice {}
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
/// use [CudaDevice::reclaim_sync()]. This will
/// perform necessary synchronization to ensure
/// that the device data finishes copying over.
///
/// # Mutating device data
///
/// This can only be done by launching kernels via
/// [crate::driver::LaunchAsync] which is implemented
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

unsafe impl<T: DeviceRepr> DeviceRepr for CudaSlice<T> {}
unsafe impl<T: Sync> Sync for CudaSlice<T> {}

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

impl<T: DeviceRepr> CudaSlice<T> {
    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn clone_async(&self) -> Result<Self, result::DriverError> {
        let dst = unsafe { self.device.alloc(self.len) }?;
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

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.clone_async().unwrap()
    }
}

impl<T: Clone + Default + DeviceRepr + Unpin> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = result::DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.device.clone().reclaim_sync(value)
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

unsafe impl DeviceRepr for CudaModule {}
unsafe impl Sync for CudaModule {}

/// Wrapper around [sys::CUfunction]. Used by [crate::driver::LaunchAsync].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    pub(crate) device: Arc<CudaDevice>,
}

unsafe impl DeviceRepr for CudaFunction {}
unsafe impl Sync for CudaFunction {}

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
    pub fn auto_joining_stream(self: &Arc<Self>) -> Result<CudaStream, result::DriverError> {
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
    pub fn join_async(self: &Arc<Self>, stream: CudaStream) -> Result<(), result::DriverError> {
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

/// A immutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice()].
///
/// See module docstring for more details.
#[allow(unused)]
pub struct CudaView<'a, T> {
    pub(crate) slice: &'a CudaSlice<T>,
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
}

/// A mutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice_mut()].
///
/// See module docstring for more details.
#[allow(unused)]
pub struct CudaViewMut<'a, T> {
    pub(crate) slice: &'a mut CudaSlice<T>,
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_helper() {
        assert_eq!((..2usize).bounds(0..=usize::MAX), Some((0, 1)));
        assert_eq!((1..2usize).bounds(..=usize::MAX), Some((1, 1)));
        assert_eq!((..).bounds(1..10), Some((1, 9)));
        assert_eq!((2..=2usize).bounds(0..=usize::MAX), Some((2, 2)));
        assert_eq!((2..=2usize).bounds(0..=1), None);
        assert_eq!((2..2usize).bounds(0..=usize::MAX), None);
    }
}
