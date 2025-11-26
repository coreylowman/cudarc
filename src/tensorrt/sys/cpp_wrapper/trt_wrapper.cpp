#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <cstring>

using namespace nvinfer1;
using namespace nvonnxparser;

class ConsoleLogger : public ILogger {
private:
    Severity minSeverity;

public:
    ConsoleLogger() : minSeverity(Severity::kWARNING) {}

    void setMinSeverity(Severity severity) {
        minSeverity = severity;
    }

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > minSeverity) return;

        const char* severityStr = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: severityStr = "INTERNAL_ERROR"; break;
            case Severity::kERROR: severityStr = "ERROR"; break;
            case Severity::kWARNING: severityStr = "WARNING"; break;
            case Severity::kINFO: severityStr = "INFO"; break;
            case Severity::kVERBOSE: severityStr = "VERBOSE"; break;
        }
        std::cerr << "[TRT " << severityStr << "] " << msg << std::endl;
    }
};

static ConsoleLogger gLogger;

extern "C" {

IBuilder* trt_create_builder() {
    return createInferBuilder(gLogger);
}

void trt_destroy_builder(IBuilder* builder) {
    if (builder) delete builder;
}

INetworkDefinition* trt_builder_create_network(IBuilder* builder, uint32_t flags) {
    if (!builder) return nullptr;
    return builder->createNetworkV2(flags);
}

IBuilderConfig* trt_builder_create_config(IBuilder* builder) {
    if (!builder) return nullptr;
    return builder->createBuilderConfig();
}

IHostMemory* trt_builder_build_serialized_network(
    IBuilder* builder,
    INetworkDefinition* network,
    IBuilderConfig* config
) {
    if (!builder || !network || !config) return nullptr;
    return builder->buildSerializedNetwork(*network, *config);
}

void trt_builder_set_max_batch_size(IBuilder* builder, int32_t batchSize) {
    (void)builder;
    (void)batchSize;
}

void trt_destroy_network(INetworkDefinition* network) {
    if (network) delete network;
}

ITensor* trt_network_add_input(
    INetworkDefinition* network,
    const char* name,
    DataType dtype,
    const Dims* dims
) {
    if (!network || !name || !dims) return nullptr;
    return network->addInput(name, dtype, *dims);
}

void trt_network_mark_output(INetworkDefinition* network, ITensor* tensor) {
    if (network && tensor) {
        network->markOutput(*tensor);
    }
}

int32_t trt_network_get_nb_inputs(INetworkDefinition* network) {
    return network ? network->getNbInputs() : 0;
}

int32_t trt_network_get_nb_outputs(INetworkDefinition* network) {
    return network ? network->getNbOutputs() : 0;
}

void trt_destroy_config(IBuilderConfig* config) {
    if (config) delete config;
}

void trt_config_set_memory_pool_limit(IBuilderConfig* config, MemoryPoolType poolType, size_t poolSize) {
    if (config) {
        config->setMemoryPoolLimit(poolType, poolSize);
    }
}

const void* trt_host_memory_data(IHostMemory* mem) {
    return mem ? mem->data() : nullptr;
}

size_t trt_host_memory_size(IHostMemory* mem) {
    return mem ? mem->size() : 0;
}

void trt_destroy_host_memory(IHostMemory* mem) {
    if (mem) delete mem;
}

IRuntime* trt_create_runtime() {
    return createInferRuntime(gLogger);
}

void trt_destroy_runtime(IRuntime* runtime) {
    if (runtime) delete runtime;
}

ICudaEngine* trt_runtime_deserialize_engine(
    IRuntime* runtime,
    const void* blob,
    size_t size
) {
    if (!runtime || !blob || size == 0) return nullptr;
    return runtime->deserializeCudaEngine(blob, size);
}

void trt_destroy_engine(ICudaEngine* engine) {
    if (engine) delete engine;
}

IExecutionContext* trt_engine_create_context(ICudaEngine* engine) {
    if (!engine) return nullptr;
    return engine->createExecutionContext();
}

int32_t trt_engine_get_nb_bindings(ICudaEngine* engine) {
    return engine ? engine->getNbIOTensors() : 0;
}

const char* trt_engine_get_tensor_name(ICudaEngine* engine, int32_t index) {
    if (!engine || index < 0) return nullptr;
    return engine->getIOTensorName(index);
}

bool trt_engine_is_execution_binding(ICudaEngine* engine, const char* name) {
    return engine && name && (engine->getTensorIOMode(name) != TensorIOMode::kNONE);
}

Dims trt_engine_get_tensor_shape(ICudaEngine* engine, const char* name) {
    if (engine && name) {
        return engine->getTensorShape(name);
    }
    return Dims{0, {0}};
}

void trt_destroy_context(IExecutionContext* context) {
    if (context) delete context;
}

bool trt_context_execute_v2(IExecutionContext* context, void** bindings) {
    if (!context || !bindings) return false;
    return context->executeV2(bindings);
}

bool trt_context_set_tensor_address(IExecutionContext* context, const char* name, void* data) {
    if (!context || !name) return false;
    return context->setTensorAddress(name, data);
}

bool trt_context_enqueue_v3(IExecutionContext* context, void* stream) {
    if (!context) return false;
    return context->enqueueV3((cudaStream_t)stream);
}

Dims trt_make_dims(int32_t nbDims, const int32_t* d) {
    Dims dims;
    dims.nbDims = nbDims;
    for (int i = 0; i < nbDims && i < Dims::MAX_DIMS; i++) {
        dims.d[i] = d[i];
    }
    return dims;
}

void trt_set_log_level(int32_t level) {
    ILogger::Severity severity;
    switch (level) {
        case 0: severity = ILogger::Severity::kINTERNAL_ERROR; break;
        case 1: severity = ILogger::Severity::kERROR; break;
        case 2: severity = ILogger::Severity::kWARNING; break;
        case 3: severity = ILogger::Severity::kINFO; break;
        case 4: severity = ILogger::Severity::kVERBOSE; break;
        default: severity = ILogger::Severity::kWARNING; break;
    }
    gLogger.setMinSeverity(severity);
}

IOptimizationProfile* trt_create_optimization_profile(IBuilder* builder) {
    if (!builder) return nullptr;
    return builder->createOptimizationProfile();
}

void trt_destroy_optimization_profile(IOptimizationProfile* profile) {
    // Note: IOptimizationProfile is owned by IBuilder and should not be deleted manually
    // This function exists for API consistency but does nothing
    (void)profile;
}

bool trt_profile_set_shape(
    IOptimizationProfile* profile,
    const char* input_name,
    const Dims* min,
    const Dims* opt,
    const Dims* max
) {
    if (!profile || !input_name || !min || !opt || !max) return false;

    return profile->setDimensions(input_name, OptProfileSelector::kMIN, *min) &&
           profile->setDimensions(input_name, OptProfileSelector::kOPT, *opt) &&
           profile->setDimensions(input_name, OptProfileSelector::kMAX, *max);
}

int32_t trt_config_add_optimization_profile(IBuilderConfig* config, IOptimizationProfile* profile) {
    if (!config || !profile) return -1;
    return config->addOptimizationProfile(profile);
}

IParser* trt_onnx_create_parser(INetworkDefinition* network) {
    if (!network) return nullptr;
    return nvonnxparser::createParser(*network, gLogger);
}

void trt_onnx_destroy_parser(IParser* parser) {
    if (parser) delete parser;
}

bool trt_onnx_parse_from_file(IParser* parser, const char* path) {
    if (!parser || !path) return false;
    return parser->parseFromFile(path, static_cast<int>(ILogger::Severity::kWARNING));
}

bool trt_onnx_parse(IParser* parser, const void* data, size_t size) {
    if (!parser || !data || size == 0) return false;
    return parser->parse(data, size);
}

int32_t trt_onnx_get_nb_errors(IParser* parser) {
    return parser ? parser->getNbErrors() : 0;
}

const char* trt_onnx_get_error_desc(IParser* parser, int32_t index) {
    if (!parser || index < 0 || index >= parser->getNbErrors()) return nullptr;
    IParserError const* error = parser->getError(index);
    return error ? error->desc() : nullptr;
}

int32_t trt_onnx_get_error_code(IParser* parser, int32_t index) {
    if (!parser || index < 0 || index >= parser->getNbErrors()) return -1;
    IParserError const* error = parser->getError(index);
    return error ? static_cast<int32_t>(error->code()) : -1;
}

int32_t trt_onnx_get_error_node(IParser* parser, int32_t index) {
    if (!parser || index < 0 || index >= parser->getNbErrors()) return -1;
    IParserError const* error = parser->getError(index);
    return error ? error->node() : -1;
}

}
