use crate::driver::{
    result::{self, DriverError},
    sys::{self, lib, CUfunc_cache_enum, CUfunction_attribute_enum},
};

use super::{alloc::DeviceRepr, device_ptr::DeviceSlice};

use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    string::String,
};

#[cfg(feature = "no-std")]
use spin::RwLock;
#[cfg(not(feature = "no-std"))]
use std::sync::RwLock;

use std::{collections::BTreeMap, marker::Unpin, sync::Arc, vec::Vec};

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
    pub fn empty_event(
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
        unsafe { result::event::elapsed(self.cu_event, other.cu_event) }
    }

    /// Returns `true` if all recorded work has been completed, `false` otherwise.
    pub fn is_complete(&self) -> bool {
        unsafe { result::event::query(self.cu_event) }.is_ok()
    }
}

/// A wrapper around [sys::CUdevice], [sys::CUcontext], [sys::CUstream],
/// and [CudaFunction].
///
/// ```rust
/// # use cudarc::driver::CudaDevice;
/// let dev = CudaDevice::new(0).unwrap();
/// ```
///
/// # Safety
/// 1. impl [Drop] to call all the corresponding resource cleanup methods
/// 2. Doesn't impl clone, so you can't have multiple device pointers
///    hanging around.
/// 3. Any allocations enforce that self is an [Arc], meaning no allocation
///    can outlive the [CudaDevice]
#[derive(Debug)]
pub struct CudaDevice {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) modules: RwLock<BTreeMap<String, Arc<CudaModule>>>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Creates a new [CudaDevice] on device index `ordinal`.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, result::DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Arc::new(CudaDevice {
            stream,
            modules: RwLock::new(BTreeMap::new()),
        }))
    }

    /// Creates a new [CudaDevice] on device index `ordinal` on a **non-default stream**.
    pub fn new_with_stream(ordinal: usize) -> Result<Arc<Self>, result::DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.new_stream()?;
        Ok(Arc::new(CudaDevice {
            stream,
            modules: RwLock::new(BTreeMap::new()),
        }))
    }

    pub fn count() -> Result<i32, result::DriverError> {
        CudaContext::device_count()
    }

    /// Get the `ordinal` index of this [CudaDevice].
    pub fn ordinal(&self) -> usize {
        self.stream.ctx.ordinal
    }

    /// Get the name of this device.
    pub fn name(&self) -> Result<String, result::DriverError> {
        self.stream.ctx.name()
    }

    /// Get the UUID of this device.
    pub fn uuid(&self) -> Result<sys::CUuuid, result::DriverError> {
        self.stream.ctx.uuid()
    }

    /// Get the underlying [sys::CUdevice] of this [CudaDevice].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the device pointer**, as it is still
    /// owned by the [CudaDevice].
    pub fn cu_device(&self) -> &sys::CUdevice {
        &self.stream.ctx.cu_device
    }

    /// Get the underlying [sys::CUcontext] of this [CudaDevice].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the context pointer**, as it is still
    /// owned by the [CudaDevice].
    pub fn cu_primary_ctx(&self) -> &sys::CUcontext {
        &self.stream.ctx.cu_ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    /// Get the underlying [sys::CUstream] that this [CudaDevice] executes
    /// all of its work on.
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the stream pointer**, as it is still
    /// owned by the [CudaDevice].
    pub fn cu_stream(&self) -> &sys::CUstream {
        &self.stream.cu_stream
    }

    /// Get the value of the specified attribute of this [CudaDevice].
    pub fn attribute(&self, attrib: sys::CUdevice_attribute) -> Result<i32, result::DriverError> {
        self.stream.ctx.attribute(attrib)
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

impl<T: Clone + Default + DeviceRepr + Unpin> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = result::DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.stream.memcpy_dtov(&value)
    }
}

/// Wrapper around [sys::CUmodule] that also contains
/// the loaded [CudaFunction] associated with this module.
///
/// See [CudaModule::get_fn()] for retrieving function handles.
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: BTreeMap<String, sys::CUfunction>,
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

/// Wrapper around [sys::CUfunction]. Used by [crate::driver::LaunchAsync].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
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
        unsafe {
            result::function::set_function_attribute(self.cu_function, attribute, value)?;
        }

        Ok(())
    }

    /// Set the cache config of this [CudaFunction].
    pub fn set_function_cache_config(
        &self,
        attribute: CUfunc_cache_enum,
    ) -> Result<(), result::DriverError> {
        unsafe {
            result::function::set_function_cache_config(self.cu_function, attribute)?;
        }

        Ok(())
    }
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

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
        let event = self.ctx.empty_event(flags)?;
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

impl CudaDevice {
    /// Allocates a new stream that can execute kernels concurrently to the default stream.
    ///
    /// The synchronization with default stream happens in **code order**. See [CudaStream] docstring.
    ///
    /// This stream synchronizes in the following way:
    /// 1. On creation it adds a wait for any existing work on the default work stream to complete
    /// 2. On drop it adds a wait for any existign work on Self to complete *to the default stream*.
    pub fn fork_default_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, result::DriverError> {
        self.stream.fork()
    }

    /// Forces [CudaStream] to drop, causing the default work stream to block on `stream`'s completion.
    /// **This is asynchronous with respect to the host.**
    pub fn wait_for(self: &Arc<Self>, stream: &CudaStream) -> Result<(), result::DriverError> {
        self.stream.wait(&stream.record_event(None)?)
    }
}

impl CudaStream {
    /// Records the current default stream's workload, and then causes `self`
    /// to wait for the default stream to finish that recorded workload.
    pub fn wait_for_default(&self) -> Result<(), result::DriverError> {
        let default_stream = self.context().default_stream();
        self.wait(&default_stream.record_event(None)?)
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
    pub fn slice(&self, range: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [CudaSlice::slice()].
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaView {
            ptr: self.cu_device_ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
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
            ptr: self.cu_device_ptr,
            len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
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
    /// # use cudarc::driver::safe::{CudaDevice, CudaSlice, CudaView};
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
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
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
            ptr: self.ptr,
            len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
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
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(range).unwrap()
    }

    /// Fallible version of [CudaSlice::slice_mut]
    pub fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaViewMut {
            ptr: self.cu_device_ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
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
            ptr: self.cu_device_ptr,
            len,
            read: &self.read,
            write: &mut self.write,
            stream: &self.stream,
            marker: PhantomData,
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
        if mid > self.len() {
            return None;
        }
        Some((
            CudaViewMut {
                ptr: self.cu_device_ptr,
                len: mid,
                read: &self.read,
                write: &self.write,
                stream: &self.stream,
                marker: PhantomData,
            },
            CudaViewMut {
                ptr: self.cu_device_ptr + (mid * std::mem::size_of::<T>()) as u64,
                len: self.len - mid,
                read: &self.read,
                write: &self.write,
                stream: &self.stream,
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
    pub fn slice<'b: 'a>(&'b self, range: impl RangeBounds<usize>) -> CudaView<'a, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice]
    pub fn try_slice<'b: 'a>(&'b self, range: impl RangeBounds<usize>) -> Option<CudaView<'a, T>> {
        range.bounds(..self.len()).map(|(start, end)| CudaView {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
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
            ptr: self.ptr,
            len,
            read: self.read,
            write: self.write,
            stream: self.stream,
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
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
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
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
            CudaViewMut {
                ptr: self.ptr + (mid * std::mem::size_of::<T>()) as u64,
                len: self.len - mid,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        ))
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
            ptr: self.ptr,
            len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
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
}
