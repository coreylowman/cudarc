use crate::driver::{
    result::{self, DriverError},
    sys::{self, lib, CUfunc_cache_enum, CUfunction_attribute_enum},
};

use std::{
    ffi::CString,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    string::String,
    sync::Arc,
    vec::Vec,
};

#[derive(Debug, PartialEq, Eq)]
pub struct CudaContext {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_ctx: sys::CUcontext,
    pub(crate) ordinal: usize,
    pub(crate) has_async_alloc: bool,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        self.bind_to_thread().unwrap();
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            unsafe { result::primary_ctx::release(self.cu_device) }.unwrap();
        }
    }
}

impl CudaContext {
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        let cu_ctx = unsafe { result::primary_ctx::retain(cu_device) }?;
        let has_async_alloc = unsafe {
            let memory_pools_supported = result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )?;
            memory_pools_supported > 0
        };
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc,
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    pub fn device_count() -> Result<i32, DriverError> {
        result::init()?;
        result::device::get_count()
    }

    /// Get the `ordinal` index of the device this is on.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the name of this device.
    pub fn name(&self) -> Result<String, result::DriverError> {
        result::device::get_name(self.cu_device)
    }

    /// Get the UUID of this device.
    pub fn uuid(&self) -> Result<sys::CUuuid, result::DriverError> {
        result::device::get_uuid(self.cu_device)
    }

    /// Get the underlying [sys::CUdevice] of this [CudaContext].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the device pointer**, as it is still
    /// owned by the [CudaContext].
    pub fn cu_device(&self) -> sys::CUdevice {
        self.cu_device
    }

    /// Get the underlying [sys::CUcontext] of this [CudaContext].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the context pointer**, as it is still
    /// owned by the [CudaContext].
    pub fn cu_ctx(&self) -> sys::CUcontext {
        self.cu_ctx
    }

    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        unsafe { result::ctx::set_current(self.cu_ctx) }
    }

    /// Get the value of the specified attribute of this [CudaDevice].
    pub fn attribute(&self, attrib: sys::CUdevice_attribute) -> Result<i32, result::DriverError> {
        unsafe { result::device::get_attribute(self.cu_device, attrib) }
    }

    /// Synchronize this context. Will only block CPU if you call [CudaContext::set_flags()] with
    /// [sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC].
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        result::ctx::synchronize()
    }

    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000"
    )))]
    pub fn set_blocking_synchronize(&self) -> Result<(), DriverError> {
        self.set_flags(sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)
    }

    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000"
    )))]
    pub fn set_flags(&self, flags: sys::CUctx_flags) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        result::ctx::set_flags(flags)
    }
}

#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) cu_event: sys::CUevent,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        self.ctx.bind_to_thread().unwrap();
        unsafe { result::event::destroy(self.cu_event) }.unwrap()
    }
}

impl CudaContext {
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let flags = flags.unwrap_or(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
        self.bind_to_thread()?;
        let cu_event = result::event::create(flags)?;
        Ok(CudaEvent {
            cu_event,
            ctx: self.clone(),
        })
    }
}

impl CudaEvent {
    pub fn cu_event(&self) -> sys::CUevent {
        self.cu_event
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx != stream.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        unsafe { result::event::record(self.cu_event, stream.cu_stream) }
    }

    /// Will only block CPU thraed if [sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC] was used to create this event.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::event::synchronize(self.cu_event) }
    }

    pub fn elapsed_ms(&self, other: &Self) -> Result<f32, DriverError> {
        if self.ctx != other.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        self.synchronize()?;
        other.synchronize()?;
        unsafe { result::event::elapsed(self.cu_event, other.cu_event) }
    }

    /// Returns `true` if all recorded work has been completed, `false` otherwise.
    pub fn is_complete(&self) -> bool {
        unsafe { result::event::query(self.cu_event) }.is_ok()
    }
}

/// A wrapper around [sys::CUstream] that safely ensures null stream is synchronized
/// upon the completion of this stream's work.
///
/// Create with [CudaDevice::fork_default_stream].
///
/// The synchronization happens in **code order**. E.g.
/// ```ignore
/// let stream = dev.fork_default_stream()?; // 0
/// dev.launch(...)?; // 1
/// function_1.launch_on_stream(&stream, ...)?; // 2
/// function_2.launch(...)?; // 3
/// drop(stream); // 4
/// dev.launch(...) // 5
/// ```
///
/// - 0 will place a streamWaitEvent(default work stream) on the new stream
/// - 1 will launch on the default work stream
/// - 2 will launch concurrently to 1 on `&stream`,
/// - 3 will launch after 1 on the default work stream, but potentially concurrently to 2.
/// - 4 will place a streamWaitEvent(`&stream`) on the default work stream
/// - 5 will happen on the default stream **after the default stream waits for 2**
///
/// **This is asynchronous with respect to the host.**
///
/// # Example with wait_for_default and wait_for
///
/// There is also a way to synchronize work on non-default streams without dropping them.
/// It can be interesting to reuse the [CudaStream] streams during the whole
/// duration of the computation.
/// To that end, one can use [CudaStream::wait_for_default] and [CudaDevice::wait_for].
///
/// Let's suppose that there are 4 streams: stream_1, stream_2, stream_3, stream_4.
/// There is no work queued on the default stream. All the work is queued on non-default streams.
/// And let's suppose that the stream dependencies are as follows:
/// - stream_1 dependencies: []
/// - stream_2 dependencies: []
/// - stream_3 dependencies: [stream_1, stream_2]
/// - stream_4 dependencies: [stream_3]
///
/// Pseudo-code:
/// ```ignore
/// let stream_1 = dev.fork_default_stream()?; // 0
/// let stream_2 = dev.fork_default_stream()?; // 1
/// let stream_3 = dev.fork_default_stream()?; // 2
/// let stream_4 = dev.fork_default_stream()?; // 3
///
/// function_1.launch_on_stream(&stream_1, ...)?; // 4
/// function_2.launch_on_stream(&stream_2, ...)?; // 5
/// dev.wait_for(&stream_1); // 6
/// dev.wait_for(&stream_2); // 7
///
/// stream_3.wait_for_default(); // 8
/// function_3.launch_on_stream(&stream_3, ...)?; // 10
/// dev.wait_for(&stream_3); // 11
///
/// stream_4.wait_for_default(); // 12
/// function_4.launch_on_stream(&stream_4, ...)?; // 13
/// dev.wait_for(&stream_4); // 14
/// ```
///
/// - function_1 and function_2 will be executed concurrently.
/// - function_3 will be executed once function_1 and function_2 have completed their execution.
/// - function_4 will be executed once function_3 has completed its execution.
///
/// This is handy because it can be use to do out-of-order execution of kernels using dependency
/// analysis. For example, with multi-head attention with 12 heads, each head can be executed
/// in its own stream.
///
/// **This is asynchronous with respect to the host.**
///
/// See [CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
/// See [3. Stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
/// See [6.6. Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
/// See [Out-of-order execution](https://en.wikipedia.org/wiki/Out-of-order_execution)
/// See [Dependence analysis](https://en.wikipedia.org/wiki/Dependence_analysis)
#[derive(Debug, PartialEq, Eq)]
pub struct CudaStream {
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        self.ctx.bind_to_thread().unwrap();
        if !self.cu_stream.is_null() {
            unsafe { result::stream::destroy(self.cu_stream).unwrap() };
        }
    }
}

impl CudaContext {
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            cu_stream: std::ptr::null_mut(),
            ctx: self.clone(),
        })
    }

    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        self.bind_to_thread()?;
        let cu_stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;
        Ok(Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
        }))
    }
}

impl CudaStream {
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        let stream = self.ctx.new_stream()?;
        stream.wait(&self.record_event(None)?)?;
        Ok(stream)
    }

    pub fn cu_stream(&self) -> sys::CUstream {
        self.cu_stream
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Will only block CPU if you call [CudaContext::set_flags()] with
    /// [sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC].
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::synchronize(self.cu_stream) }
    }

    pub fn record_event(
        &self,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let event = self.ctx.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }

    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        if self.ctx != event.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        unsafe {
            result::stream::wait_event(
                self.cu_stream,
                event.cu_event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
    }

    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
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
    pub(crate) read: CudaEvent,
    pub(crate) write: CudaEvent,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) marker: PhantomData<*const T>,
}

unsafe impl<T> Send for CudaSlice<T> {}
unsafe impl<T> Sync for CudaSlice<T> {}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        self.stream.wait(&self.read).unwrap();
        self.stream.wait(&self.write).unwrap();
        unsafe { result::free_async(self.cu_device_ptr, self.stream.cu_stream) }.unwrap();
    }
}

impl<T> CudaSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn ordinal(&self) -> usize {
        self.stream.ctx.ordinal
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.ctx
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn try_clone(&self) -> Result<Self, result::DriverError> {
        self.stream.clone_dtod(self)
    }
}

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: Clone + Default + DeviceRepr> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = result::DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.stream.memcpy_dtov(&value)
    }
}

/// A immutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice()] or [CudaSlice::slice()].
///
/// This type is to [CudaSlice] as `&[T]` is to `Vec<T>`.
#[derive(Debug)]
pub struct CudaView<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: &'a CudaEvent,
    pub(crate) write: &'a CudaEvent,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a [T]>,
}

impl<T> CudaSlice<T> {
    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            ptr: self.cu_device_ptr,
            len: self.len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        }
    }
}

impl<T> CudaView<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn resize(&self, start: usize, end: usize) -> Self {
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }
}

/// A mutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice_mut()] or [CudaSlice::slice_mut()].
///
/// This type is to [CudaSlice] as `&mut [T]` is to `Vec<T>`.
#[derive(Debug)]
pub struct CudaViewMut<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: &'a CudaEvent,
    pub(crate) write: &'a CudaEvent,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a mut [T]>,
}

impl<T> CudaSlice<T> {
    pub fn as_view_mut(&self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            ptr: self.cu_device_ptr,
            len: self.len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        }
    }
}

impl<T> CudaViewMut<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            ptr: self.ptr,
            len: self.len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    fn resize(&self, start: usize, end: usize) -> Self {
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
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

/// Something that can be copied to device memory and
/// turned into a parameter for [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because a struct should likely
/// be `#[repr(C)]` to be represented in cuda memory,
/// and not all types are valid.
pub unsafe trait DeviceRepr {}
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

pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stream(&self) -> &Arc<CudaStream>;
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T>: DeviceSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr;
    fn read_event(&self) -> &CudaEvent;
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError>;
    fn record_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.read_event().record(stream)
    }
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.cu_device_ptr
    }
    fn read_event(&self) -> &CudaEvent {
        &self.read
    }
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(&self.write)
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
    fn read_event(&self) -> &CudaEvent {
        self.read
    }
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(self.write)
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
    fn read_event(&self) -> &CudaEvent {
        self.read
    }
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(self.write)
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T>: DevicePtr<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr;
    fn write_event(&self) -> &CudaEvent;
    fn block_for_write(&self, stream: &CudaStream) -> Result<(), DriverError>;
    fn record_write(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.write_event().record(stream)
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.cu_device_ptr
    }
    fn write_event(&self) -> &CudaEvent {
        &self.write
    }
    fn block_for_write(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(&self.read)?;
        stream.wait(&self.write)
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.ptr
    }
    fn write_event(&self) -> &CudaEvent {
        self.write
    }
    fn block_for_write(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(self.read)?;
        stream.wait(self.write)
    }
}

pub trait HostSlice<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`. Otherwise
    /// You may run into device synchronization errors
    unsafe fn stream_synced_slice(&self, stream: &CudaStream) -> Result<&[T], DriverError>;

    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`. Otherwise
    /// You may run into device synchronization errors
    unsafe fn stream_synced_mut_slice(
        &mut self,
        stream: &CudaStream,
    ) -> Result<&mut [T], DriverError>;

    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError>;
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
    unsafe fn stream_synced_slice(&self, _stream: &CudaStream) -> Result<&[T], DriverError> {
        Ok(self)
    }
    unsafe fn stream_synced_mut_slice(
        &mut self,
        _stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        Ok(self)
    }
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.synchronize()
    }
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn stream_synced_slice(&self, _stream: &CudaStream) -> Result<&[T], DriverError> {
        Ok(self)
    }
    unsafe fn stream_synced_mut_slice(
        &mut self,
        _stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        Ok(self)
    }
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.synchronize()
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn stream_synced_slice(&self, _stream: &CudaStream) -> Result<&[T], DriverError> {
        Ok(self)
    }
    unsafe fn stream_synced_mut_slice(
        &mut self,
        _stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        Ok(self)
    }
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.synchronize()
    }
}

#[derive(Debug)]
pub struct PinnedHostSlice<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: CudaEvent,
}

impl<T> Drop for PinnedHostSlice<T> {
    fn drop(&mut self) {
        self.event.synchronize().unwrap();
        unsafe { result::free_host(self.ptr as _) }.unwrap();
    }
}

impl CudaContext {
    /// Allocates page locked host memory with [sys::CU_MEMHOSTALLOC_WRITECOMBINED] flags.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)
    ///
    /// # Safety
    /// 1. This is unsafe because the memory is unset after this call.
    pub unsafe fn alloc_pinned<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<PinnedHostSlice<T>, DriverError> {
        self.bind_to_thread()?;
        let ptr = result::malloc_host(
            len * std::mem::size_of::<T>(),
            sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        )?;
        let ptr = ptr as *mut T;
        assert!(!ptr.is_null());
        assert!(len * std::mem::size_of::<T>() < isize::MAX as usize);
        assert!(ptr.is_aligned());
        let event = self.new_event(None)?;
        Ok(PinnedHostSlice { ptr, len, event })
    }
}

impl<T> PinnedHostSlice<T> {
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.event.ctx
    }

    /// The size of the slice
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ValidAsZeroBits> PinnedHostSlice<T> {
    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_ptr(&self) -> Result<*const T, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_ptr(&mut self) -> Result<*mut T, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_slice(&self) -> Result<&[T], DriverError> {
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], DriverError> {
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}

impl<T> HostSlice<T> for PinnedHostSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice(&self, stream: &CudaStream) -> Result<&[T], DriverError> {
        stream.wait(&self.event)?;
        Ok(std::slice::from_raw_parts(self.ptr, self.len))
    }

    unsafe fn stream_synced_mut_slice(
        &mut self,
        stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        stream.wait(&self.event)?;
        Ok(std::slice::from_raw_parts_mut(self.ptr, self.len))
    }

    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.event.record(stream)
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
        let read = self.ctx.new_event(None)?;
        let write = self.ctx.new_event(None)?;
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
        let read = self.ctx.new_event(None)?;
        let write = self.ctx.new_event(None)?;
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
        assert!(dst.len() >= src.len());
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
        assert!(dst.len() >= src.len());
        let dst = unsafe { dst.stream_synced_mut_slice(self) }?;
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
        assert!(dst.len() >= src.len());
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

impl<T> CudaSlice<T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice(0..50);
    /// do_something(&view);
    /// ```
    ///
    /// Like a normal slice, borrow checking prevents the underlying [CudaSlice] from being dropped.
    /// ```rust,compile_fail
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaView};
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
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// Fallible version of [CudaSlice::slice()].
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        self.as_view().try_slice(bounds)
    }

    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// do_something(&mut view);
    /// ```
    ///
    /// Like a normal mutable slice, borrow checking prevents the underlying [CudaSlice] from being dropped.
    /// ```rust,compile_fail
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
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
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view1 = slice.slice_mut(0..50);
    /// // cannot borrow twice from slice
    /// let mut view2 = slice.slice_mut(50..100);
    /// do_something(view1, view2);
    /// ```
    /// If you need non-overlapping mutable views into a [CudaSlice], you can use [CudaSlice::split_at_mut()].
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [CudaSlice::slice_mut]
    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        to_range(bounds, self.len).map(|(start, end)| self.as_view_mut().resize(start, end))
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.cu_device_ptr,
                len,
                read: &self.read,
                write: &self.write,
                stream: &self.stream,
                marker: PhantomData,
            },
        )
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaViewMut {
                ptr: self.cu_device_ptr,
                len,
                read: &self.read,
                write: &mut self.write,
                stream: &self.stream,
                marker: PhantomData,
            },
        )
    }

    pub fn split_at(&self, mid: usize) -> (CudaView<'_, T>, CudaView<'_, T>) {
        self.try_split_at(mid).unwrap()
    }

    /// Fallible version of [CudaSlice::split_at].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at(&self, mid: usize) -> Option<(CudaView<'_, T>, CudaView<'_, T>)> {
        (mid <= self.len()).then(|| {
            let view = self.as_view();
            (view.resize(0, mid), view.resize(mid, self.len))
        })
    }

    /// Splits the [CudaSlice] into two at the given index, returning two [CudaViewMut] for the two halves.
    ///
    /// Panics if `mid > self.len`.
    ///
    /// This method can be used to create non-overlapping mutable views into a [CudaSlice].
    /// ```rust
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
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
        (mid <= self.len()).then(|| {
            let view = self.as_view_mut();
            (view.resize(0, mid), view.resize(mid, self.len))
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
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice(0..50);
    /// let mut view2 = view.slice(0..25);
    /// do_something(&view);
    /// ```
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'a, T> {
        self.try_slice(bounds).unwrap()
    }

    /// Fallible version of [CudaView::slice]
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'a, T>> {
        to_range(bounds, self.len).map(|(start, end)| self.resize(start, end))
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.ptr,
                len,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        self.try_split_at(mid).unwrap()
    }

    /// Fallible version of [CudaSlice::split_at].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at(&self, mid: usize) -> Option<(Self, Self)> {
        (mid <= self.len()).then(|| (self.resize(0, mid), self.resize(mid, self.len)))
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
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
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
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
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
    pub fn slice<'b: 'a>(&'b self, bounds: impl RangeBounds<usize>) -> CudaView<'a, T> {
        self.try_slice(bounds).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice]
    pub fn try_slice<'b: 'a>(&'b self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'a, T>> {
        to_range(bounds, self.len).map(|(start, end)| self.as_view().resize(start, end))
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.ptr,
                len,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Self {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice_mut]
    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<Self> {
        to_range(bounds, self.len).map(|(start, end)| self.resize(start, end))
    }

    /// Splits the [CudaViewMut] into two at the given index.
    ///
    /// Panics if `mid > self.len`.
    ///
    /// This method can be used to create non-overlapping mutable views into a [CudaViewMut].
    /// ```rust
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let dev = CudaDevice::new(0).unwrap();
    /// let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// // split the view into two non-overlapping, mutable views
    /// let (mut view1, mut view2) = view.split_at_mut(25);
    /// do_something(view1, view2);
    pub fn split_at_mut(&mut self, mid: usize) -> (Self, Self) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [CudaViewMut::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`
    pub fn try_split_at_mut(&mut self, mid: usize) -> Option<(Self, Self)> {
        (mid <= self.len()).then(|| (self.resize(0, mid), self.resize(mid, self.len)))
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'a, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaViewMut {
                ptr: self.ptr,
                len,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }
}

fn to_range(range: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n + 1,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    (end <= len).then_some((start, end))
}

/// Wrapper around [sys::CUmodule] that also contains
/// the loaded [CudaFunction] associated with this module.
///
/// See [CudaModule::get_fn()] for retrieving function handles.
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.bind_to_thread().unwrap();
        unsafe { result::module::unload(self.cu_module) }.unwrap();
    }
}

impl CudaContext {
    /// Dynamically load a set of [crate::driver::CudaFunction] from a jit compiled ptx.
    ///
    /// - `ptx` contains the compiled ptx
    /// - `func_names` is a slice of function names to load into the module during build.
    pub fn load_module(
        self: &Arc<Self>,
        ptx: crate::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>, result::DriverError> {
        self.bind_to_thread()?;

        let cu_module = match ptx.0 {
            crate::nvrtc::PtxKind::Image(image) => unsafe {
                result::module::load_data(image.as_ptr() as *const _)
            },
            crate::nvrtc::PtxKind::Src(src) => {
                let c_src = CString::new(src).unwrap();
                unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
            }
            crate::nvrtc::PtxKind::File(path) => {
                let name_c = CString::new(path.to_str().unwrap()).unwrap();
                result::module::load(name_c)
            }
        }?;
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
}

/// Wrapper around [sys::CUfunction]. Used by [crate::driver::LaunchAsync].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        let fn_name_c = CString::new(fn_name).unwrap();
        let cu_function = unsafe { result::module::get_function(self.cu_module, fn_name_c) }?;
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }
}

impl CudaFunction {
    pub fn occupancy_available_dynamic_smem_per_block(
        &self,
        num_blocks: u32,
        block_size: u32,
    ) -> Result<usize, result::DriverError> {
        let mut dynamic_smem_size: usize = 0;

        unsafe {
            lib()
                .cuOccupancyAvailableDynamicSMemPerBlock(
                    &mut dynamic_smem_size,
                    self.cu_function,
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
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> Result<u32, result::DriverError> {
        let mut num_blocks: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);

        unsafe {
            lib()
                .cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &mut num_blocks,
                    self.cu_function,
                    block_size as std::ffi::c_int,
                    dynamic_smem_size,
                    flags as std::ffi::c_uint,
                )
                .result()?
        };

        Ok(num_blocks as u32)
    }

    #[cfg(not(any(
        feature = "cuda-11070",
        feature = "cuda-11060",
        feature = "cuda-11050",
        feature = "cuda-11040"
    )))]
    pub fn occupancy_max_active_clusters(
        &self,
        config: crate::driver::LaunchConfig,
        stream: &CudaStream,
    ) -> Result<u32, result::DriverError> {
        let mut num_clusters: std::ffi::c_int = 0;

        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: stream.cu_stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            lib()
                .cuOccupancyMaxActiveClusters(&mut num_clusters, self.cu_function, &cfg)
                .result()?
        };

        Ok(num_clusters as u32)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        block_size_to_dynamic_smem_size: extern "C" fn(block_size: std::ffi::c_int) -> usize,
        dynamic_smem_size: usize,
        block_size_limit: u32,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> Result<(u32, u32), result::DriverError> {
        let mut min_grid_size: std::ffi::c_int = 0;
        let mut block_size: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);

        unsafe {
            lib()
                .cuOccupancyMaxPotentialBlockSizeWithFlags(
                    &mut min_grid_size,
                    &mut block_size,
                    self.cu_function,
                    Some(block_size_to_dynamic_smem_size),
                    dynamic_smem_size,
                    block_size_limit as std::ffi::c_int,
                    flags as std::ffi::c_uint,
                )
                .result()?
        };

        Ok((min_grid_size as u32, block_size as u32))
    }

    #[cfg(not(any(
        feature = "cuda-11070",
        feature = "cuda-11060",
        feature = "cuda-11050",
        feature = "cuda-11040"
    )))]
    pub fn occupancy_max_potential_cluster_size(
        &self,
        config: crate::driver::LaunchConfig,
        stream: &CudaStream,
    ) -> Result<u32, result::DriverError> {
        let mut cluster_size: std::ffi::c_int = 0;

        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: stream.cu_stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            lib()
                .cuOccupancyMaxPotentialClusterSize(&mut cluster_size, self.cu_function, &cfg)
                .result()?
        };

        Ok(cluster_size as u32)
    }

    /// Set the value of a specific attribute of this [CudaFunction].
    pub fn set_attribute(
        &self,
        attribute: CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), result::DriverError> {
        unsafe { result::function::set_function_attribute(self.cu_function, attribute, value) }
    }

    /// Set the cache config of this [CudaFunction].
    pub fn set_function_cache_config(
        &self,
        attribute: CUfunc_cache_enum,
    ) -> Result<(), result::DriverError> {
        unsafe { result::function::set_function_cache_config(self.cu_function, attribute) }
    }
}

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
        let read = self.ctx.new_event(None).unwrap();
        let write = self.ctx.new_event(None).unwrap();
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_transmutes() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
        assert!(unsafe { slice.transmute::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute::<f32>(26) }.is_none());
        assert!(unsafe { slice.transmute_mut::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute_mut::<f32>(26) }.is_none());

        {
            let view = slice.slice(0..100);
            assert!(unsafe { view.transmute::<f32>(25) }.is_some());
            assert!(unsafe { view.transmute::<f32>(26) }.is_none());
        }

        {
            let mut view_mut = slice.slice_mut(0..100);
            assert!(unsafe { view_mut.transmute::<f32>(25) }.is_some());
            assert!(unsafe { view_mut.transmute::<f32>(26) }.is_none());
            assert!(unsafe { view_mut.transmute_mut::<f32>(25) }.is_some());
            assert!(unsafe { view_mut.transmute_mut::<f32>(26) }.is_none());
        }
    }

    #[test]
    fn test_threading() {
        let ctx1 = CudaContext::new(0).unwrap();
        let ctx2 = ctx1.clone();

        let thread1 = std::thread::spawn(move || {
            ctx1.bind_to_thread()?;
            ctx1.default_stream().alloc_zeros::<f32>(10)
        });
        let thread2 = std::thread::spawn(move || {
            ctx2.bind_to_thread()?;
            ctx2.default_stream().alloc_zeros::<f32>(10)
        });

        let _: crate::driver::CudaSlice<f32> = thread1.join().unwrap().unwrap();
        let _: crate::driver::CudaSlice<f32> = thread2.join().unwrap().unwrap();
    }

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
        assert_eq!(Arc::strong_count(&ctx), 2);
        assert_eq!(Arc::strong_count(&stream), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&ctx), 2);
        assert_eq!(Arc::strong_count(&stream), 1);
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
