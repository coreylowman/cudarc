//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaContext], [CudaStream], and more.

pub(crate) mod core;
pub(crate) mod external_memory;
pub(crate) mod graph;
pub(crate) mod launch;
pub(crate) mod profile;
pub(crate) mod unified_memory;

pub use self::core::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CuDevicePtr, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, HostSlice, PinnedHostSlice, SyncOnDrop,
    ValidAsZeroBits,
};
pub use self::external_memory::{ExternalMemory, MappedBuffer};
pub use self::graph::CudaGraph;
pub use self::launch::{LaunchArgs, LaunchConfig, PushKernelArg};
pub use self::profile::{profiler_start, profiler_stop, Profiler};
pub use self::unified_memory::UnifiedSlice;
pub use crate::driver::result::DriverError;
