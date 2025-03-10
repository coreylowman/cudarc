//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaDevice], [CudaStream], and more.

pub(crate) mod alloc;
pub(crate) mod core;
pub(crate) mod device_ptr;
pub(crate) mod external_memory;
pub(crate) mod host_slice;
pub(crate) mod launch;
pub(crate) mod profile;
pub(crate) mod ptx;

pub use self::alloc::{DeviceRepr, ValidAsZeroBits};
pub use self::core::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
};
pub use self::device_ptr::{DevicePtr, DevicePtrMut, DeviceSlice};
pub use self::external_memory::{ExternalMemory, MappedBuffer};
pub use self::host_slice::{HostSlice, PinnedHostSlice};
pub use self::launch::{LaunchArgs, LaunchConfig, PushKernelArg};
pub use self::profile::{profiler_start, profiler_stop, Profiler};

pub use crate::driver::result::DriverError;
