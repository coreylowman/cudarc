use crate::runtime::{
    result,
    sys::{self, lib},
};
use core::ffi::c_void;

use super::{alloc::DeviceRepr, device_ptr::DeviceSlice};

use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

use std::{marker::Unpin, pin::Pin, sync::Arc, vec::Vec};

/// A wrapper around [sys::CUdevice], [sys::CUcontext], [sys::CUstream],
/// and [CudaFunction].
///
/// ```rust
/// # use cudarc::runtime::CudaDevice;
/// let dev = CudaDevice::new(0).unwrap();
/// ```
///
/// # Safety
/// 1. impl [Drop] to call all the corresponding resource cleanup methods
/// 2. Doesn't impl clone, so you can't have multiple device pointers
/// hanging around.
/// 3. Any allocations enforce that self is an [Arc], meaning no allocation
/// can outlive the [CudaDevice]
#[derive(Debug)]
pub struct CudaDevice {
    pub(crate) device_prop: sys::cudaDeviceProp,
    /// The stream that all work is executed on.
    pub(crate) stream: sys::cudaStream_t,
    /// Used to synchronize with stream
    pub(crate) event: sys::cudaEvent_t,
    pub(crate) ordinal: usize,
    pub(crate) is_async: bool,
    /// In order to prevent the primary context created by the driver API from being destroyed
    pub(crate) _driver_device: Option<Arc<crate::driver::CudaDevice>>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Creates a new [CudaDevice] on device index `ordinal`.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, result::RuntimeError> {
        result::device::set(ordinal as i32)?;

        let device_prop = result::device::get_device_prop(ordinal as i32)?;

        // can fail with OOM
        let event = result::event::create(sys::cudaEventDisableTiming)?;

        let value = unsafe {
            result::device::get_attribute(
                ordinal as i32,
                sys::cudaDeviceAttr::cudaDevAttrMemoryPoolsSupported,
            )?
        };

        let is_async = value > 0;

        let device = CudaDevice {
            device_prop,
            stream: std::ptr::null_mut(),
            event,
            ordinal,
            is_async,
            _driver_device: None,
        };
        Ok(Arc::new(device))
    }

    /// Creates a [CudaDevice] from [crate::driver::CudaDevice].
    pub fn from_driver(
        device: &Arc<crate::driver::CudaDevice>,
    ) -> Result<Arc<Self>, result::RuntimeError> {
        device.bind_to_thread().unwrap();
        result::device::set(device.ordinal() as i32)?;
        Ok(Arc::new(CudaDevice {
            device_prop: result::device::get_device_prop(device.ordinal() as i32)?,
            stream: device.stream as sys::cudaStream_t,
            event: device.event as sys::cudaEvent_t,
            ordinal: device.ordinal(),
            is_async: device.is_async,
            _driver_device: Some(device.clone()),
        }))
    }

    /// Creates a new [CudaDevice] on device index `ordinal` on a **non-default stream**.
    pub fn new_with_stream(ordinal: usize) -> Result<Arc<Self>, result::RuntimeError> {
        result::device::set(ordinal as i32)?;

        let device_prop = result::device::get_device_prop(ordinal as i32)?;

        // can fail with OOM
        let event = result::event::create(sys::cudaEventDisableTiming)?;

        let value = unsafe {
            result::device::get_attribute(
                ordinal as i32,
                sys::cudaDeviceAttr::cudaDevAttrMemoryPoolsSupported,
            )?
        };

        let is_async = value > 0;

        let stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;

        let device = CudaDevice {
            device_prop,
            stream,
            event,
            ordinal,
            is_async,
            _driver_device: None,
        };

        Ok(Arc::new(device))
    }

    pub fn count() -> Result<i32, result::RuntimeError> {
        result::device::get_count()
    }

    /// Get the `ordinal` index of this [CudaDevice].
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the underlying [sys::cudaDeviceProp] of this [CudaDevice].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the device pointer**, as it is still
    /// owned by the [CudaDevice].
    pub fn device_prop(&self) -> &sys::cudaDeviceProp {
        &self.device_prop
    }

    /// Get the underlying [sys::cudaStream_t] that this [CudaDevice] executes
    /// all of its work on.
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the stream pointer**, as it is still
    /// owned by the [CudaDevice].
    pub fn stream(&self) -> &sys::cudaStream_t {
        &self.stream
    }

    /// Get the value of the specified attribute of this [CudaDevice].
    pub fn attribute(&self, attrib: sys::cudaDeviceAttr) -> Result<i32, result::RuntimeError> {
        unsafe { result::device::get_attribute(self.ordinal as i32, attrib) }
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // we will let the driver api handle the cleanup
        if self._driver_device.is_none() {
            // Synchronize and destroy the stream if it exists.
            let stream = std::mem::replace(&mut self.stream, std::ptr::null_mut());
            if !stream.is_null() {
                unsafe { result::stream::synchronize(stream) }.unwrap();
                unsafe { result::stream::destroy(stream) }.unwrap();
            }

            // Destroy the event if it exists.
            let event = std::mem::replace(&mut self.event, std::ptr::null_mut());
            if !event.is_null() {
                unsafe { result::event::destroy(event) }.unwrap();
            }
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
/// use [CudaDevice::sync_reclaim()]. This will
/// perform necessary synchronization to ensure
/// that the device data finishes copying over.
///
/// # Mutating device data
///
/// This can only be done by launching kernels via
/// [crate::runtime::LaunchAsync] which is implemented
/// by [CudaDevice]. Pass `&mut CudaSlice<T>`
/// if you want to mutate the rc, and `&CudaSlice<T>` otherwise.
///
/// Unfortunately, `&CudaSlice<T>` can **still be mutated
/// by the [CudaFunction]**.
#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) device_ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) _driver_device: Option<Arc<crate::driver::CudaDevice>>,
    pub(crate) host_buf: Option<Pin<Vec<T>>>,
}

unsafe impl<T: Send> Send for CudaSlice<T> {}
unsafe impl<T: Sync> Sync for CudaSlice<T> {}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        unsafe {
            if self.device.is_async {
                result::free_async(self.device_ptr as *mut c_void, self.device.stream).unwrap();
            } else {
                result::free_sync(self.device_ptr as *mut c_void).unwrap();
            }
        }
    }
}

impl<T> CudaSlice<T> {
    /// Get a clone of the underlying [CudaDevice].
    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn try_clone(&self) -> Result<Self, result::RuntimeError> {
        let mut dst = unsafe { self.device.alloc(self.len) }?;
        self.device.dtod_copy(self, &mut dst)?;
        Ok(dst)
    }
}

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: Clone + Default + DeviceRepr + Unpin> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = result::RuntimeError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.device.clone().sync_reclaim(value)
    }
}

/// Wrapper around [sys::cudaFunction_t]. Used by [crate::runtime::LaunchAsync].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cuda_function: sys::cudaFunction_t,
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) _driver_device: Option<Arc<crate::driver::CudaDevice>>,
}

pub enum CudaOccupancyFlagsEnum {
    OccupancyDefault = sys::cudaOccupancyDefault as isize,
    OccupancyDisableCachingOverride = sys::cudaOccupancyDisableCachingOverride as isize,
}

impl CudaFunction {
    pub fn new(device: &Arc<CudaDevice>, cuda_function: sys::cudaFunction_t) -> Self {
        if let Some(driver_device) = &device._driver_device {
            driver_device.bind_to_thread().unwrap();
        }
        CudaFunction {
            cuda_function,
            device: device.clone(),
            _driver_device: device._driver_device.clone(),
        }
    }

    pub fn occupancy_available_dynamic_smem_per_block(
        &self,
        num_blocks: u32,
        block_size: u32,
    ) -> Result<usize, result::RuntimeError> {
        let mut dynamic_smem_size: usize = 0;

        unsafe {
            lib()
                .cudaOccupancyAvailableDynamicSMemPerBlock(
                    &mut dynamic_smem_size,
                    self.cuda_function as *const c_void,
                    num_blocks as std::ffi::c_int,
                    block_size as std::ffi::c_int,
                )
                .result()?
        };

        Ok(dynamic_smem_size)
    }

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_smem_size: usize,
        flags: Option<CudaOccupancyFlagsEnum>,
    ) -> Result<u32, result::RuntimeError> {
        let mut num_blocks: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(CudaOccupancyFlagsEnum::OccupancyDefault);

        unsafe {
            lib()
                .cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &mut num_blocks,
                    self.cuda_function as *const c_void,
                    block_size as std::ffi::c_int,
                    dynamic_smem_size,
                    flags as std::ffi::c_uint,
                )
                .result()?
        };

        Ok(num_blocks as u32)
    }

    #[cfg(not(feature = "cuda-11070"))]
    pub fn occupancy_max_active_clusters(
        &self,
        config: crate::runtime::LaunchConfig,
        shared_mem_size: u32,
    ) -> Result<u32, result::RuntimeError> {
        let mut num_clusters: std::ffi::c_int = 0;

        let cfg = sys::cudaLaunchConfig_st {
            gridDim: sys::dim3 {
                x: config.grid_dim.0,
                y: config.grid_dim.1,
                z: config.grid_dim.2,
            },
            blockDim: sys::dim3 {
                x: config.block_dim.0,
                y: config.block_dim.1,
                z: config.block_dim.2,
            },
            dynamicSmemBytes: shared_mem_size as usize,
            stream: self.device.stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            lib()
                .cudaOccupancyMaxActiveClusters(
                    &mut num_clusters,
                    self.cuda_function as *const c_void,
                    &cfg,
                )
                .result()?
        };

        Ok(num_clusters as u32)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        block_size_to_dynamic_smem_size: extern "C" fn(block_size: std::ffi::c_int) -> usize,
        dynamic_smem_size: usize,
        block_size_limit: u32,
        flags: Option<crate::driver::sys::CUoccupancy_flags_enum>,
    ) -> Result<(u32, u32), crate::driver::result::DriverError> {
        let mut min_grid_size: std::ffi::c_int = 0;
        let mut block_size: std::ffi::c_int = 0;
        let flags =
            flags.unwrap_or(crate::driver::sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);

        unsafe {
            crate::driver::sys::lib()
                .cuOccupancyMaxPotentialBlockSizeWithFlags(
                    &mut min_grid_size,
                    &mut block_size,
                    self.cuda_function as crate::driver::sys::CUfunction,
                    Some(block_size_to_dynamic_smem_size),
                    dynamic_smem_size,
                    block_size_limit as std::ffi::c_int,
                    flags as std::ffi::c_uint,
                )
                .result()?
        };

        Ok((min_grid_size as u32, block_size as u32))
    }

    #[cfg(not(feature = "cuda-11070"))]
    pub fn occupancy_max_potential_cluster_size(
        &self,
        config: crate::runtime::LaunchConfig,
        shared_mem_size: u32,
    ) -> Result<u32, result::RuntimeError> {
        let mut cluster_size: std::ffi::c_int = 0;

        let cfg = sys::cudaLaunchConfig_st {
            gridDim: sys::dim3 {
                x: config.grid_dim.0,
                y: config.grid_dim.1,
                z: config.grid_dim.2,
            },
            blockDim: sys::dim3 {
                x: config.block_dim.0,
                y: config.block_dim.1,
                z: config.block_dim.2,
            },
            dynamicSmemBytes: shared_mem_size as usize,
            stream: self.device.stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            lib()
                .cudaOccupancyMaxPotentialClusterSize(
                    &mut cluster_size,
                    self.cuda_function as *const c_void,
                    &cfg,
                )
                .result()?
        };

        Ok(cluster_size as u32)
    }

    /// Set the value of a specific attribute of this [CudaFunction].
    pub fn set_attribute(
        &self,
        attribute: sys::cudaFuncAttribute,
        value: i32,
    ) -> Result<(), result::RuntimeError> {
        unsafe {
            result::function::set_function_attribute(
                self.cuda_function as *const c_void,
                attribute,
                value,
            )?;
        }

        Ok(())
    }
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

/// Wrapper around [sys::cudaStream_t] that safely ensures null stream is synchronized
/// upon the completion of this streams work.
///
/// Create with [CudaDevice::fork_default_stream].
///
/// The synchronization happens in **code order**. E.g.
/// ```ignore
/// let stream = dev.fork_default_stream()?; // 0
/// dev.launch(...)?; // 1
/// dev.launch_on_stream(&stream, ...)?; // 2
/// dev.launch(...)?; // 3
/// drop(stream); // 4
/// dev.launch(...) // 5
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
    pub stream: sys::cudaStream_t,
    device: Arc<CudaDevice>,
}

impl CudaDevice {
    /// Allocates a new stream that can execute kernels concurrently to the default stream.
    ///
    /// The synchronization with default stream happens in **code order**. See [CudaStream] docstring.
    ///
    /// This stream synchronizes in the following way:
    /// 1. On creation it adds a wait for any existing work on the default work stream to complete
    /// 2. On drop it adds a wait for any existign work on Self to complete *to the default stream*.
    pub fn fork_default_stream(self: &Arc<Self>) -> Result<CudaStream, result::RuntimeError> {
        let stream = CudaStream {
            stream: result::stream::create(result::stream::StreamKind::NonBlocking)?,
            device: self.clone(),
        };
        stream.wait_for_default()?;
        Ok(stream)
    }

    /// Forces [CudaStream] to drop, causing the default work stream to block on `streams` completion.
    /// **This is asynchronous with respect to the host.**
    #[allow(unused_variables)]
    pub fn wait_for(self: &Arc<Self>, stream: &CudaStream) -> Result<(), result::RuntimeError> {
        unsafe {
            result::event::record(self.event, stream.stream)?;
            result::stream::wait_event(self.stream, self.event, sys::cudaEventWaitDefault)
        }
    }
}

impl CudaStream {
    /// Record's the current default streams workload, and then causes `self`
    /// to wait for the default stream to finish that recorded workload.
    pub fn wait_for_default(&self) -> Result<(), result::RuntimeError> {
        unsafe {
            result::event::record(self.device.event, self.device.stream)?;
            result::stream::wait_event(self.stream, self.device.event, sys::cudaEventWaitDefault)
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        self.device.wait_for(self).unwrap();
        unsafe {
            result::stream::destroy(self.stream).unwrap();
        }
    }
}

/// A immutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice()] or [CudaSlice::slice()].
///
/// This type is to [CudaSlice] as `&[T]` is to `Vec<T>`.
#[derive(Debug)]
pub struct CudaView<'a, T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    marker: PhantomData<&'a [T]>,
}

impl<T> CudaSlice<T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice(0..50);
    /// do_something(&view);
    /// ```
    ///
    /// Like a normal slice, borrow checking prevents the underlying [CudaSlice] from being dropped.
    /// ```rust,compile_fail
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let view = {
    ///     let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    ///     let mut view = slice.slice(0..50);
    ///     // cannot return view, since it borrows from slice
    ///     view
    /// };
    /// do_something(&view);
    /// ```
    pub fn slice(&self, range: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [CudaSlice::slice()].
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaView {
            ptr: unsafe { self.device_ptr.add(start) },
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.num_bytes()).then_some(CudaView {
            ptr: self.device_ptr as *mut S,
            len,
            marker: PhantomData,
        })
    }
}

impl<'a, T> CudaView<'a, T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice(0..50);
    /// let mut view2 = view.slice(0..25);
    /// do_something(&view);
    /// ```
    pub fn slice(&self, range: impl RangeBounds<usize>) -> CudaView<'a, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [CudaView::slice]
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<CudaView<'a, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaView {
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            marker: PhantomData,
        })
    }
}

/// A mutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice_mut()] or [CudaSlice::slice_mut()].
///
/// This type is to [CudaSlice] as `&mut [T]` is to `Vec<T>`.
#[derive(Debug)]
pub struct CudaViewMut<'a, T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    marker: PhantomData<&'a mut [T]>,
}

impl<T> CudaSlice<T> {
    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// do_something(&mut view);
    /// ```
    ///
    /// Like a normal mutable slice, borrow checking prevents the underlying [CudaSlice] from being dropped.
    /// ```rust,compile_fail
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut view = {
    ///     let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    ///     let view = slice.slice_mut(0..50);
    ///     // cannot return view, since it borrows from slice
    ///     view
    /// };
    /// do_something(&mut view);
    /// ```
    ///
    /// Like with normal mutable slices, one cannot mutably slice twice into the same [CudaSlice]:
    /// ```rust,compile_fail
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view1 = slice.slice_mut(0..50);
    /// // cannot borrow twice from slice
    /// let mut view2 = slice.slice_mut(50..100);
    /// do_something(view1, view2);
    /// ```
    /// If you need non-overlapping mutable views into a [CudaSlice], you can use [CudaSlice::split_at_mut()].
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(range).unwrap()
    }

    /// Fallible version of [CudaSlice::slice_mut]
    pub fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaViewMut {
            ptr: unsafe { self.device_ptr.add(start) },
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.num_bytes()).then_some(CudaViewMut {
            ptr: self.device_ptr as *mut S,
            len,
            marker: PhantomData,
        })
    }

    /// Splits the [CudaSlice] into two at the given index, returning two [CudaViewMut] for the two halves.
    ///
    /// Panics if `mid > self.len`.
    ///
    /// This method can be used to create non-overlapping mutable views into a [CudaSlice].
    /// ```rust
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// // split the slice into two non-overlapping, mutable views
    /// let (mut view1, mut view2) = slice.split_at_mut(50);
    /// do_something(view1, view2);
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'_, T>, CudaViewMut<'_, T>) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [CudaSlice::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'_, T>, CudaViewMut<'_, T>)> {
        if mid > self.len() {
            return None;
        }
        Some((
            CudaViewMut {
                ptr: self.device_ptr,
                len: mid,
                marker: PhantomData,
            },
            CudaViewMut {
                ptr: unsafe { self.device_ptr.add(mid) },
                len: self.len - mid,
                marker: PhantomData,
            },
        ))
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// let mut view2 = view.slice_mut(0..25);
    /// do_something(&mut view2);
    /// ```
    ///
    /// One cannot slice twice into the same [CudaViewMut]:
    /// ```rust,compile_fail
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// // cannot borrow twice from same view
    /// let mut view1 = slice.slice_mut(0..25);
    /// let mut view2 = slice.slice_mut(25..50);
    /// do_something(view1, view2);
    /// ```
    /// If you need non-overlapping mutable views into a [CudaViewMut], you can use [CudaViewMut::split_at_mut()].
    pub fn slice<'b: 'a>(&'b self, range: impl RangeBounds<usize>) -> CudaView<'a, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice]
    pub fn try_slice<'b: 'a>(&'b self, range: impl RangeBounds<usize>) -> Option<CudaView<'a, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaView {
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    pub fn slice_mut<'b: 'a>(&'b mut self, range: impl RangeBounds<usize>) -> CudaViewMut<'a, T> {
        self.try_slice_mut(range).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice_mut]
    pub fn try_slice_mut<'b: 'a>(
        &'b mut self,
        range: impl RangeBounds<usize>,
    ) -> Option<CudaViewMut<'a, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaViewMut {
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Splits the [CudaViewMut] into two at the given index.
    ///
    /// Panics if `mid > self.len`.
    ///
    /// This method can be used to create non-overlapping mutable views into a [CudaViewMut].
    /// ```rust
    /// # use cudarc::runtime::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// // split the view into two non-overlapping, mutable views
    /// let (mut view1, mut view2) = view.split_at_mut(25);
    /// do_something(view1, view2);
    pub fn split_at_mut<'b: 'a>(
        &'b mut self,
        mid: usize,
    ) -> (CudaViewMut<'a, T>, CudaViewMut<'a, T>) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [CudaViewMut::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`
    pub fn try_split_at_mut<'b: 'a>(
        &'b mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'a, T>, CudaViewMut<'a, T>)> {
        if mid > self.len() {
            return None;
        }
        Some((
            CudaViewMut {
                ptr: self.ptr,
                len: mid,
                marker: PhantomData,
            },
            CudaViewMut {
                ptr: unsafe { self.ptr.add(mid) },
                len: self.len - mid,
                marker: PhantomData,
            },
        ))
    }
}

trait RangeHelper: RangeBounds<usize> {
    fn inclusive_start(&self, valid_start: usize) -> usize;
    fn exclusive_end(&self, valid_end: usize) -> usize;
    fn bounds(&self, valid: impl RangeHelper) -> Option<(usize, usize)> {
        let vs = valid.inclusive_start(0);
        let ve = valid.exclusive_end(usize::MAX);
        let s = self.inclusive_start(vs);
        let e = self.exclusive_end(ve);

        let inside = s >= vs && e <= ve;
        let valid = s < e || (s == e && !matches!(self.end_bound(), Bound::Included(_)));

        (inside && valid).then_some((s, e))
    }
}
impl<R: RangeBounds<usize>> RangeHelper for R {
    fn inclusive_start(&self, valid_start: usize) -> usize {
        match self.start_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) => *n + 1,
            Bound::Unbounded => valid_start,
        }
    }
    fn exclusive_end(&self, valid_end: usize) -> usize {
        match self.end_bound() {
            Bound::Included(n) => *n + 1,
            Bound::Excluded(n) => *n,
            Bound::Unbounded => valid_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_bounds_helper() {
        assert_eq!((..2usize).bounds(0..usize::MAX), Some((0, 2)));
        assert_eq!((1..2usize).bounds(..usize::MAX), Some((1, 2)));
        assert_eq!((..).bounds(1..10), Some((1, 10)));
        assert_eq!((2..=2usize).bounds(0..usize::MAX), Some((2, 3)));
        assert_eq!((2..=2usize).bounds(0..=1), None);
        assert_eq!((2..2usize).bounds(0..usize::MAX), Some((2, 2)));
        assert_eq!((1..0usize).bounds(0..usize::MAX), None);
        assert_eq!((1..=0usize).bounds(0..usize::MAX), None);
    }

    #[test]
    fn test_transmutes() {
        let dev = CudaDevice::new(0).unwrap();
        let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
        assert!(unsafe { slice.transmute::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute::<f32>(26) }.is_none());
        assert!(unsafe { slice.transmute_mut::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute_mut::<f32>(26) }.is_none());
    }
}
