#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum DataType {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum LogSeverity {
    kINTERNAL_ERROR = 0,
    kERROR = 1,
    kWARNING = 2,
    kINFO = 3,
    kVERBOSE = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum MemoryPoolType {
    kWORKSPACE = 0,
    kDLA_MANAGED_SRAM = 1,
    kDLA_LOCAL_DRAM = 2,
    kDLA_GLOBAL_DRAM = 3,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Dims {
    pub nbDims: i32,
    pub d: [i64; 8usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IBuilder {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IBuilderConfig {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ICudaEngine {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IExecutionContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IHostMemory {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct INetworkDefinition {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IOnnxParser {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IOptimizationProfile {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IParserError {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IRuntime {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ITensor {
    _unused: [u8; 0],
}
// TensorRT wrapper is always statically linked, so always use direct extern declarations
extern "C" {
    pub fn trt_builder_build_serialized_network(
        builder: *mut IBuilder,
        network: *mut INetworkDefinition,
        config: *mut IBuilderConfig,
    ) -> *mut IHostMemory;
    pub fn trt_builder_create_config(builder: *mut IBuilder) -> *mut IBuilderConfig;
    pub fn trt_builder_create_network(
        builder: *mut IBuilder,
        flags: u32,
    ) -> *mut INetworkDefinition;
    pub fn trt_builder_set_max_batch_size(builder: *mut IBuilder, batchSize: i32);
    pub fn trt_config_add_optimization_profile(
        config: *mut IBuilderConfig,
        profile: *mut IOptimizationProfile,
    ) -> i32;
    pub fn trt_config_set_memory_pool_limit(
        config: *mut IBuilderConfig,
        poolType: MemoryPoolType,
        poolSize: usize,
    );
    pub fn trt_context_enqueue_v3(
        context: *mut IExecutionContext,
        stream: *mut ::core::ffi::c_void,
    ) -> bool;
    pub fn trt_context_execute_v2(
        context: *mut IExecutionContext,
        bindings: *mut *mut ::core::ffi::c_void,
    ) -> bool;
    pub fn trt_context_set_tensor_address(
        context: *mut IExecutionContext,
        name: *const ::core::ffi::c_char,
        data: *mut ::core::ffi::c_void,
    ) -> bool;
    pub fn trt_create_builder() -> *mut IBuilder;
    pub fn trt_create_optimization_profile(builder: *mut IBuilder) -> *mut IOptimizationProfile;
    pub fn trt_create_runtime() -> *mut IRuntime;
    pub fn trt_destroy_builder(builder: *mut IBuilder);
    pub fn trt_destroy_config(config: *mut IBuilderConfig);
    pub fn trt_destroy_context(context: *mut IExecutionContext);
    pub fn trt_destroy_engine(engine: *mut ICudaEngine);
    pub fn trt_destroy_host_memory(mem: *mut IHostMemory);
    pub fn trt_destroy_network(network: *mut INetworkDefinition);
    pub fn trt_destroy_optimization_profile(profile: *mut IOptimizationProfile);
    pub fn trt_destroy_runtime(runtime: *mut IRuntime);
    pub fn trt_engine_create_context(engine: *mut ICudaEngine) -> *mut IExecutionContext;
    pub fn trt_engine_get_nb_bindings(engine: *mut ICudaEngine) -> i32;
    pub fn trt_engine_get_tensor_name(
        engine: *mut ICudaEngine,
        index: i32,
    ) -> *const ::core::ffi::c_char;
    pub fn trt_engine_get_tensor_shape(
        engine: *mut ICudaEngine,
        name: *const ::core::ffi::c_char,
    ) -> Dims;
    pub fn trt_engine_is_execution_binding(
        engine: *mut ICudaEngine,
        name: *const ::core::ffi::c_char,
    ) -> bool;
    pub fn trt_host_memory_data(mem: *mut IHostMemory) -> *const ::core::ffi::c_void;
    pub fn trt_host_memory_size(mem: *mut IHostMemory) -> usize;
    pub fn trt_make_dims(nbDims: i32, d: *const i32) -> Dims;
    pub fn trt_network_add_input(
        network: *mut INetworkDefinition,
        name: *const ::core::ffi::c_char,
        dtype: DataType,
        dims: *const Dims,
    ) -> *mut ITensor;
    pub fn trt_network_get_nb_inputs(network: *mut INetworkDefinition) -> i32;
    pub fn trt_network_get_nb_outputs(network: *mut INetworkDefinition) -> i32;
    pub fn trt_network_mark_output(network: *mut INetworkDefinition, tensor: *mut ITensor);
    pub fn trt_onnx_create_parser(network: *mut INetworkDefinition) -> *mut IOnnxParser;
    pub fn trt_onnx_destroy_parser(parser: *mut IOnnxParser);
    pub fn trt_onnx_get_error_code(parser: *mut IOnnxParser, index: i32) -> i32;
    pub fn trt_onnx_get_error_desc(
        parser: *mut IOnnxParser,
        index: i32,
    ) -> *const ::core::ffi::c_char;
    pub fn trt_onnx_get_error_node(parser: *mut IOnnxParser, index: i32) -> i32;
    pub fn trt_onnx_get_nb_errors(parser: *mut IOnnxParser) -> i32;
    pub fn trt_onnx_parse(
        parser: *mut IOnnxParser,
        data: *const ::core::ffi::c_void,
        size: usize,
    ) -> bool;
    pub fn trt_onnx_parse_from_file(
        parser: *mut IOnnxParser,
        path: *const ::core::ffi::c_char,
    ) -> bool;
    pub fn trt_profile_set_shape(
        profile: *mut IOptimizationProfile,
        input_name: *const ::core::ffi::c_char,
        min: *const Dims,
        opt: *const Dims,
        max: *const Dims,
    ) -> bool;
    pub fn trt_runtime_deserialize_engine(
        runtime: *mut IRuntime,
        blob: *const ::core::ffi::c_void,
        size: usize,
    ) -> *mut ICudaEngine;
    pub fn trt_set_log_level(level: LogSeverity);
}
