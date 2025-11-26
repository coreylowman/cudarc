// TensorRT C Wrapper Header
// C-style interface to TensorRT C++ API

#ifndef TRT_WRAPPER_H
#define TRT_WRAPPER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IBuilder IBuilder;
typedef struct INetworkDefinition INetworkDefinition;
typedef struct IBuilderConfig IBuilderConfig;
typedef struct IHostMemory IHostMemory;
typedef struct IRuntime IRuntime;
typedef struct ICudaEngine ICudaEngine;
typedef struct IExecutionContext IExecutionContext;
typedef struct ITensor ITensor;

typedef enum {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
} DataType;

typedef enum {
    kWORKSPACE = 0,
    kDLA_MANAGED_SRAM = 1,
    kDLA_LOCAL_DRAM = 2,
    kDLA_GLOBAL_DRAM = 3,
} MemoryPoolType;

typedef struct {
    int32_t nbDims;
    int64_t d[8];
} Dims;

// Builder
IBuilder* trt_create_builder();
void trt_destroy_builder(IBuilder* builder);
INetworkDefinition* trt_builder_create_network(IBuilder* builder, uint32_t flags);
IBuilderConfig* trt_builder_create_config(IBuilder* builder);
IHostMemory* trt_builder_build_serialized_network(
    IBuilder* builder,
    INetworkDefinition* network,
    IBuilderConfig* config
);
void trt_builder_set_max_batch_size(IBuilder* builder, int32_t batchSize);

// Network
void trt_destroy_network(INetworkDefinition* network);
ITensor* trt_network_add_input(
    INetworkDefinition* network,
    const char* name,
    DataType dtype,
    const Dims* dims
);
void trt_network_mark_output(INetworkDefinition* network, ITensor* tensor);
int32_t trt_network_get_nb_inputs(INetworkDefinition* network);
int32_t trt_network_get_nb_outputs(INetworkDefinition* network);

// Config
void trt_destroy_config(IBuilderConfig* config);
void trt_config_set_memory_pool_limit(IBuilderConfig* config, MemoryPoolType poolType, size_t poolSize);

// HostMemory
const void* trt_host_memory_data(IHostMemory* mem);
size_t trt_host_memory_size(IHostMemory* mem);
void trt_destroy_host_memory(IHostMemory* mem);

// Runtime
IRuntime* trt_create_runtime();
void trt_destroy_runtime(IRuntime* runtime);
ICudaEngine* trt_runtime_deserialize_engine(IRuntime* runtime, const void* blob, size_t size);

// Engine
void trt_destroy_engine(ICudaEngine* engine);
IExecutionContext* trt_engine_create_context(ICudaEngine* engine);
int32_t trt_engine_get_nb_bindings(ICudaEngine* engine);
const char* trt_engine_get_tensor_name(ICudaEngine* engine, int32_t index);
bool trt_engine_is_execution_binding(ICudaEngine* engine, const char* name);
Dims trt_engine_get_tensor_shape(ICudaEngine* engine, const char* name);

// ExecutionContext
void trt_destroy_context(IExecutionContext* context);
bool trt_context_execute_v2(IExecutionContext* context, void** bindings);
bool trt_context_set_tensor_address(IExecutionContext* context, const char* name, void* data);
bool trt_context_enqueue_v3(IExecutionContext* context, void* stream);

// Helpers
Dims trt_make_dims(int32_t nbDims, const int32_t* d);

// Logger
typedef enum {
    kINTERNAL_ERROR = 0,
    kERROR = 1,
    kWARNING = 2,
    kINFO = 3,
    kVERBOSE = 4,
} LogSeverity;

void trt_set_log_level(LogSeverity level);

// Optimization Profile
typedef struct IOptimizationProfile IOptimizationProfile;

IOptimizationProfile* trt_create_optimization_profile(IBuilder* builder);
void trt_destroy_optimization_profile(IOptimizationProfile* profile);
bool trt_profile_set_shape(
    IOptimizationProfile* profile,
    const char* input_name,
    const Dims* min,
    const Dims* opt,
    const Dims* max
);
int32_t trt_config_add_optimization_profile(IBuilderConfig* config, IOptimizationProfile* profile);

// ONNX Parser
typedef struct IOnnxParser IOnnxParser;
typedef struct IParserError IParserError;

IOnnxParser* trt_onnx_create_parser(INetworkDefinition* network);
void trt_onnx_destroy_parser(IOnnxParser* parser);
bool trt_onnx_parse_from_file(IOnnxParser* parser, const char* path);
bool trt_onnx_parse(IOnnxParser* parser, const void* data, size_t size);
int32_t trt_onnx_get_nb_errors(IOnnxParser* parser);
const char* trt_onnx_get_error_desc(IOnnxParser* parser, int32_t index);
int32_t trt_onnx_get_error_code(IOnnxParser* parser, int32_t index);
int32_t trt_onnx_get_error_node(IOnnxParser* parser, int32_t index);

#ifdef __cplusplus
}
#endif

#endif
