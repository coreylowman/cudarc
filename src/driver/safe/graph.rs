use std::sync::Arc;

use crate::driver::{result, sys};

use super::{CudaStream, DriverError};

/// Represents a replay-able Cuda Graph. Create with [CudaStream::begin_capture()] and [CudaStream::end_capture()].
///
/// Once created you can replay with [CudaGraph::launch()].
///
/// # On Thread safety
///
/// This object is **NOT** thread safe.
///
/// From official docs:
///
/// > Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed concurrently from multiple threads. API calls accessing the same graph object must be serialized externally.
/// >
/// > Note that this includes APIs which may appear to be read-only, such as cudaGraphClone() (cuGraphClone()) and cudaGraphInstantiate() (cuGraphInstantiate()). No API or pair of APIs is guaranteed to be safe to call on the same graph object from two different threads without serialization.
///
/// <https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety>
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            unsafe { result::graph::exec_destroy(cu_graph_exec) }.unwrap();
        }

        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            unsafe { result::graph::destroy(cu_graph) }.unwrap();
        }
    }
}

impl CudaStream {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143)
    pub fn begin_capture(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        unsafe { result::stream::begin_capture(self.cu_stream, mode) }
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c)
    ///
    /// `flags` is passed to [cuGraphInstantiate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1)
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
        if cu_graph.is_null() {
            return Ok(None);
        }
        let cu_graph_exec = unsafe { result::graph::instantiate(cu_graph, flags) }?;
        Ok(Some(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.clone(),
        }))
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca)
    pub fn capture_status(&self) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        unsafe { result::stream::is_capturing(self.cu_stream) }
    }
}

impl CudaGraph {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    pub fn launch(&self) -> Result<(), DriverError> {
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream) }
    }
}
