//! A thin wrapper around [sys] providing [Result]s with [TensorRTError].

use super::sys;
use std::mem::MaybeUninit;

/// TensorRT error type. Since TensorRT uses boolean returns and logs errors,
/// we use string-based error messages.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorRTError(pub String);

impl TensorRTError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for TensorRTError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorRT error: {}", self.0)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TensorRTError {}

// ============================================================================
// Logger
// ============================================================================

/// Set the TensorRT log level.
///
/// # Safety
/// This function is safe to call at any time.
pub fn set_log_level(level: sys::LogSeverity) {
    unsafe {
        sys::trt_set_log_level(level);
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Create a TensorRT builder.
///
/// See [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/api/).
pub fn create_builder() -> Result<*mut sys::IBuilder, TensorRTError> {
    let builder = unsafe { sys::trt_create_builder() };
    if builder.is_null() {
        Err(TensorRTError::new("Failed to create TensorRT builder"))
    } else {
        Ok(builder)
    }
}

/// Destroy a TensorRT builder.
///
/// # Safety
/// The builder must not have already been freed.
pub unsafe fn destroy_builder(builder: *mut sys::IBuilder) {
    sys::trt_destroy_builder(builder);
}

/// Create a network definition from a builder.
///
/// # Safety
/// The builder must be valid and not freed.
pub unsafe fn builder_create_network(
    builder: *mut sys::IBuilder,
    flags: u32,
) -> Result<*mut sys::INetworkDefinition, TensorRTError> {
    let network = sys::trt_builder_create_network(builder, flags);
    if network.is_null() {
        Err(TensorRTError::new("Failed to create network definition"))
    } else {
        Ok(network)
    }
}

/// Create a builder configuration.
///
/// # Safety
/// The builder must be valid and not freed.
pub unsafe fn builder_create_config(
    builder: *mut sys::IBuilder,
) -> Result<*mut sys::IBuilderConfig, TensorRTError> {
    let config = sys::trt_builder_create_config(builder);
    if config.is_null() {
        Err(TensorRTError::new("Failed to create builder config"))
    } else {
        Ok(config)
    }
}

/// Build a serialized network.
///
/// # Safety
/// The builder, network, and config must all be valid and not freed.
pub unsafe fn builder_build_serialized_network(
    builder: *mut sys::IBuilder,
    network: *mut sys::INetworkDefinition,
    config: *mut sys::IBuilderConfig,
) -> Result<*mut sys::IHostMemory, TensorRTError> {
    let serialized = sys::trt_builder_build_serialized_network(builder, network, config);
    if serialized.is_null() {
        Err(TensorRTError::new("Failed to build serialized network"))
    } else {
        Ok(serialized)
    }
}

/// Set the maximum batch size for the builder.
///
/// # Safety
/// The builder must be valid and not freed.
pub unsafe fn builder_set_max_batch_size(builder: *mut sys::IBuilder, batch_size: i32) {
    sys::trt_builder_set_max_batch_size(builder, batch_size);
}

// ============================================================================
// Network
// ============================================================================

/// Destroy a network definition.
///
/// # Safety
/// The network must not have already been freed.
pub unsafe fn destroy_network(network: *mut sys::INetworkDefinition) {
    sys::trt_destroy_network(network);
}

/// Add an input tensor to the network.
///
/// # Safety
/// The network must be valid and not freed. The name must be a valid C string.
pub unsafe fn network_add_input(
    network: *mut sys::INetworkDefinition,
    name: *const std::ffi::c_char,
    dtype: sys::DataType,
    dims: *const sys::Dims,
) -> Result<*mut sys::ITensor, TensorRTError> {
    let tensor = sys::trt_network_add_input(network, name, dtype, dims);
    if tensor.is_null() {
        Err(TensorRTError::new("Failed to add input tensor"))
    } else {
        Ok(tensor)
    }
}

/// Mark a tensor as a network output.
///
/// # Safety
/// The network and tensor must both be valid and not freed.
pub unsafe fn network_mark_output(
    network: *mut sys::INetworkDefinition,
    tensor: *mut sys::ITensor,
) {
    sys::trt_network_mark_output(network, tensor);
}

/// Get the number of inputs in the network.
///
/// # Safety
/// The network must be valid and not freed.
pub unsafe fn network_get_nb_inputs(network: *mut sys::INetworkDefinition) -> i32 {
    sys::trt_network_get_nb_inputs(network)
}

/// Get the number of outputs in the network.
///
/// # Safety
/// The network must be valid and not freed.
pub unsafe fn network_get_nb_outputs(network: *mut sys::INetworkDefinition) -> i32 {
    sys::trt_network_get_nb_outputs(network)
}

// ============================================================================
// BuilderConfig
// ============================================================================

/// Destroy a builder configuration.
///
/// # Safety
/// The config must not have already been freed.
pub unsafe fn destroy_config(config: *mut sys::IBuilderConfig) {
    sys::trt_destroy_config(config);
}

/// Set the memory pool limit for a builder configuration.
///
/// # Safety
/// The config must be valid and not freed.
pub unsafe fn config_set_memory_pool_limit(
    config: *mut sys::IBuilderConfig,
    pool_type: sys::MemoryPoolType,
    pool_size: usize,
) {
    sys::trt_config_set_memory_pool_limit(config, pool_type, pool_size);
}

// ============================================================================
// HostMemory
// ============================================================================

/// Get a pointer to the host memory data.
///
/// # Safety
/// The memory must be valid and not freed.
pub unsafe fn host_memory_data(mem: *mut sys::IHostMemory) -> *const std::ffi::c_void {
    sys::trt_host_memory_data(mem)
}

/// Get the size of the host memory.
///
/// # Safety
/// The memory must be valid and not freed.
pub unsafe fn host_memory_size(mem: *mut sys::IHostMemory) -> usize {
    sys::trt_host_memory_size(mem)
}

/// Destroy host memory.
///
/// # Safety
/// The memory must not have already been freed.
pub unsafe fn destroy_host_memory(mem: *mut sys::IHostMemory) {
    sys::trt_destroy_host_memory(mem);
}

// ============================================================================
// Runtime
// ============================================================================

/// Create a TensorRT runtime.
pub fn create_runtime() -> Result<*mut sys::IRuntime, TensorRTError> {
    let runtime = unsafe { sys::trt_create_runtime() };
    if runtime.is_null() {
        Err(TensorRTError::new("Failed to create TensorRT runtime"))
    } else {
        Ok(runtime)
    }
}

/// Destroy a TensorRT runtime.
///
/// # Safety
/// The runtime must not have already been freed.
pub unsafe fn destroy_runtime(runtime: *mut sys::IRuntime) {
    sys::trt_destroy_runtime(runtime);
}

/// Deserialize an engine from a blob.
///
/// # Safety
/// The runtime must be valid and not freed. The blob must point to valid serialized engine data.
pub unsafe fn runtime_deserialize_engine(
    runtime: *mut sys::IRuntime,
    blob: *const std::ffi::c_void,
    size: usize,
) -> Result<*mut sys::ICudaEngine, TensorRTError> {
    let engine = sys::trt_runtime_deserialize_engine(runtime, blob, size);
    if engine.is_null() {
        Err(TensorRTError::new("Failed to deserialize engine"))
    } else {
        Ok(engine)
    }
}

// ============================================================================
// Engine
// ============================================================================

/// Destroy a CUDA engine.
///
/// # Safety
/// The engine must not have already been freed.
pub unsafe fn destroy_engine(engine: *mut sys::ICudaEngine) {
    sys::trt_destroy_engine(engine);
}

/// Create an execution context from an engine.
///
/// # Safety
/// The engine must be valid and not freed.
pub unsafe fn engine_create_context(
    engine: *mut sys::ICudaEngine,
) -> Result<*mut sys::IExecutionContext, TensorRTError> {
    let context = sys::trt_engine_create_context(engine);
    if context.is_null() {
        Err(TensorRTError::new("Failed to create execution context"))
    } else {
        Ok(context)
    }
}

/// Get the number of bindings in the engine.
///
/// # Safety
/// The engine must be valid and not freed.
pub unsafe fn engine_get_nb_bindings(engine: *mut sys::ICudaEngine) -> i32 {
    sys::trt_engine_get_nb_bindings(engine)
}

/// Get the name of a tensor by index.
///
/// # Safety
/// The engine must be valid and not freed. The index must be valid.
pub unsafe fn engine_get_tensor_name(
    engine: *mut sys::ICudaEngine,
    index: i32,
) -> *const std::ffi::c_char {
    sys::trt_engine_get_tensor_name(engine, index)
}

/// Check if a binding is an execution binding.
///
/// # Safety
/// The engine must be valid and not freed. The name must be a valid C string.
pub unsafe fn engine_is_execution_binding(
    engine: *mut sys::ICudaEngine,
    name: *const std::ffi::c_char,
) -> bool {
    sys::trt_engine_is_execution_binding(engine, name)
}

/// Get the shape of a tensor.
///
/// # Safety
/// The engine must be valid and not freed. The name must be a valid C string.
pub unsafe fn engine_get_tensor_shape(
    engine: *mut sys::ICudaEngine,
    name: *const std::ffi::c_char,
) -> sys::Dims {
    sys::trt_engine_get_tensor_shape(engine, name)
}

// ============================================================================
// ExecutionContext
// ============================================================================

/// Destroy an execution context.
///
/// # Safety
/// The context must not have already been freed.
pub unsafe fn destroy_context(context: *mut sys::IExecutionContext) {
    sys::trt_destroy_context(context);
}

/// Execute inference synchronously with the V2 API.
///
/// # Safety
/// The context must be valid and not freed. Bindings must point to valid device memory.
pub unsafe fn context_execute_v2(
    context: *mut sys::IExecutionContext,
    bindings: *mut *mut std::ffi::c_void,
) -> Result<(), TensorRTError> {
    if sys::trt_context_execute_v2(context, bindings) {
        Ok(())
    } else {
        Err(TensorRTError::new("Inference execution failed"))
    }
}

/// Set the address of a tensor binding.
///
/// # Safety
/// The context must be valid and not freed. The name must be a valid C string.
/// The data must point to valid device memory.
pub unsafe fn context_set_tensor_address(
    context: *mut sys::IExecutionContext,
    name: *const std::ffi::c_char,
    data: *mut std::ffi::c_void,
) -> Result<(), TensorRTError> {
    if sys::trt_context_set_tensor_address(context, name, data) {
        Ok(())
    } else {
        Err(TensorRTError::new("Failed to set tensor address"))
    }
}

/// Enqueue inference on a CUDA stream (V3 API).
///
/// # Safety
/// The context must be valid and not freed. The stream must be a valid CUDA stream.
pub unsafe fn context_enqueue_v3(
    context: *mut sys::IExecutionContext,
    stream: *mut std::ffi::c_void,
) -> Result<(), TensorRTError> {
    if sys::trt_context_enqueue_v3(context, stream) {
        Ok(())
    } else {
        Err(TensorRTError::new("Inference enqueue failed"))
    }
}

// ============================================================================
// Optimization Profile
// ============================================================================

/// Create an optimization profile.
///
/// # Safety
/// The builder must be valid and not freed.
pub unsafe fn create_optimization_profile(
    builder: *mut sys::IBuilder,
) -> Result<*mut sys::IOptimizationProfile, TensorRTError> {
    let profile = sys::trt_create_optimization_profile(builder);
    if profile.is_null() {
        Err(TensorRTError::new("Failed to create optimization profile"))
    } else {
        Ok(profile)
    }
}

/// Destroy an optimization profile.
///
/// # Safety
/// The profile must not have already been freed.
pub unsafe fn destroy_optimization_profile(profile: *mut sys::IOptimizationProfile) {
    sys::trt_destroy_optimization_profile(profile);
}

/// Set the shape for an input in an optimization profile.
///
/// # Safety
/// The profile must be valid and not freed. The input_name must be a valid C string.
pub unsafe fn profile_set_shape(
    profile: *mut sys::IOptimizationProfile,
    input_name: *const std::ffi::c_char,
    min: *const sys::Dims,
    opt: *const sys::Dims,
    max: *const sys::Dims,
) -> Result<(), TensorRTError> {
    if sys::trt_profile_set_shape(profile, input_name, min, opt, max) {
        Ok(())
    } else {
        Err(TensorRTError::new(
            "Failed to set optimization profile shape",
        ))
    }
}

/// Add an optimization profile to a builder configuration.
///
/// # Safety
/// The config and profile must both be valid and not freed.
pub unsafe fn config_add_optimization_profile(
    config: *mut sys::IBuilderConfig,
    profile: *mut sys::IOptimizationProfile,
) -> i32 {
    sys::trt_config_add_optimization_profile(config, profile)
}

// ============================================================================
// ONNX Parser
// ============================================================================

/// Create an ONNX parser for a network.
///
/// # Safety
/// The network must be valid and not freed.
pub unsafe fn onnx_create_parser(
    network: *mut sys::INetworkDefinition,
) -> Result<*mut sys::IOnnxParser, TensorRTError> {
    let parser = sys::trt_onnx_create_parser(network);
    if parser.is_null() {
        Err(TensorRTError::new("Failed to create ONNX parser"))
    } else {
        Ok(parser)
    }
}

/// Destroy an ONNX parser.
///
/// # Safety
/// The parser must not have already been freed.
pub unsafe fn onnx_destroy_parser(parser: *mut sys::IOnnxParser) {
    sys::trt_onnx_destroy_parser(parser);
}

/// Parse an ONNX model from a file.
///
/// # Safety
/// The parser must be valid and not freed. The path must be a valid C string pointing to an ONNX file.
pub unsafe fn onnx_parse_from_file(
    parser: *mut sys::IOnnxParser,
    path: *const std::ffi::c_char,
) -> Result<(), TensorRTError> {
    if sys::trt_onnx_parse_from_file(parser, path) {
        Ok(())
    } else {
        let nb_errors = sys::trt_onnx_get_nb_errors(parser);
        if nb_errors > 0 {
            let error_desc = std::ffi::CStr::from_ptr(sys::trt_onnx_get_error_desc(parser, 0))
                .to_string_lossy()
                .into_owned();
            Err(TensorRTError::new(format!(
                "ONNX parsing failed: {}",
                error_desc
            )))
        } else {
            Err(TensorRTError::new("ONNX parsing failed"))
        }
    }
}

/// Parse an ONNX model from a memory buffer.
///
/// # Safety
/// The parser must be valid and not freed. The data must point to valid ONNX model data.
pub unsafe fn onnx_parse(
    parser: *mut sys::IOnnxParser,
    data: *const std::ffi::c_void,
    size: usize,
) -> Result<(), TensorRTError> {
    if sys::trt_onnx_parse(parser, data, size) {
        Ok(())
    } else {
        let nb_errors = sys::trt_onnx_get_nb_errors(parser);
        if nb_errors > 0 {
            let error_desc = std::ffi::CStr::from_ptr(sys::trt_onnx_get_error_desc(parser, 0))
                .to_string_lossy()
                .into_owned();
            Err(TensorRTError::new(format!(
                "ONNX parsing failed: {}",
                error_desc
            )))
        } else {
            Err(TensorRTError::new("ONNX parsing failed"))
        }
    }
}

/// Get the number of errors from the ONNX parser.
///
/// # Safety
/// The parser must be valid and not freed.
pub unsafe fn onnx_get_nb_errors(parser: *mut sys::IOnnxParser) -> i32 {
    sys::trt_onnx_get_nb_errors(parser)
}

/// Get the description of an error from the ONNX parser.
///
/// # Safety
/// The parser must be valid and not freed. The index must be valid (< nb_errors).
pub unsafe fn onnx_get_error_desc(
    parser: *mut sys::IOnnxParser,
    index: i32,
) -> *const std::ffi::c_char {
    sys::trt_onnx_get_error_desc(parser, index)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a Dims structure from a slice of dimensions.
pub fn make_dims(dims: &[i32]) -> sys::Dims {
    let mut d = sys::Dims {
        nbDims: dims.len() as i32,
        d: [0; 8],
    };
    for (i, &dim) in dims.iter().enumerate() {
        d.d[i] = dim as i64;
    }
    unsafe { sys::trt_make_dims(d.nbDims, dims.as_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_destroy_builder() {
        // Test basic builder creation and destruction
        let builder = create_builder().expect("Failed to create builder");
        assert!(!builder.is_null(), "Builder should not be null");

        unsafe {
            destroy_builder(builder);
        }
    }

    #[test]
    fn test_create_runtime() {
        // Test runtime creation
        let runtime = create_runtime().expect("Failed to create runtime");
        assert!(!runtime.is_null(), "Runtime should not be null");

        unsafe {
            destroy_runtime(runtime);
        }
    }

    #[test]
    fn test_builder_create_network() {
        // Test creating a network from a builder
        let builder = create_builder().expect("Failed to create builder");

        unsafe {
            // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH = 1
            let network = builder_create_network(builder, 1).expect("Failed to create network");
            assert!(!network.is_null(), "Network should not be null");

            // Check network properties
            let nb_inputs = network_get_nb_inputs(network);
            let nb_outputs = network_get_nb_outputs(network);
            assert_eq!(nb_inputs, 0, "New network should have 0 inputs");
            assert_eq!(nb_outputs, 0, "New network should have 0 outputs");

            destroy_network(network);
            destroy_builder(builder);
        }
    }

    #[test]
    fn test_builder_create_config() {
        // Test creating a builder config
        let builder = create_builder().expect("Failed to create builder");

        unsafe {
            let config = builder_create_config(builder).expect("Failed to create config");
            assert!(!config.is_null(), "Config should not be null");

            // Test setting memory pool limit
            config_set_memory_pool_limit(
                config,
                sys::MemoryPoolType::kWORKSPACE,
                1 << 30, // 1GB
            );

            destroy_config(config);
            destroy_builder(builder);
        }
    }

    #[test]
    fn test_make_dims() {
        // Test dimension creation
        let dims = make_dims(&[1, 3, 224, 224]);
        assert_eq!(dims.nbDims, 4);
        assert_eq!(dims.d[0], 1);
        assert_eq!(dims.d[1], 3);
        assert_eq!(dims.d[2], 224);
        assert_eq!(dims.d[3], 224);
    }

    #[test]
    fn test_set_log_level() {
        // Test setting log level (should not crash)
        set_log_level(sys::LogSeverity::kWARNING);
        set_log_level(sys::LogSeverity::kERROR);
    }

    #[test]
    fn test_create_optimization_profile() {
        // Test optimization profile creation
        let builder = create_builder().expect("Failed to create builder");

        unsafe {
            let profile = create_optimization_profile(builder)
                .expect("Failed to create optimization profile");
            assert!(!profile.is_null(), "Profile should not be null");

            destroy_optimization_profile(profile);
            destroy_builder(builder);
        }
    }
}
