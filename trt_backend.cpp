// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trt_backend.cpp
 * @brief   TensorRT and neural network inference runtime backend
 */

#include "trt_backend.h"
#include "trtinfer_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#ifdef HAVE_CUDA
#    include <cuda_runtime.h>
#endif

#ifdef HAVE_TENSORRT
#    include <NvInfer.h>
#    include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;
#endif

enum ModelType
{
    MODEL_TYPE_UNKNOWN = 0,
    MODEL_TYPE_BIN_MLP = 1,
    MODEL_TYPE_TRT_ENGINE = 2,
    MODEL_TYPE_ONNX = 3
};

struct trt_context
{
    char      model_path[512];
    int       gpu_id;
    float     last_latency_us;
    ModelType model_type;

    trt_tensor_desc_t in_desc;
    trt_tensor_desc_t out_desc;

    /* Host buffers */
    std::vector<float> h_input;
    std::vector<float> h_output;

    /* For MODEL_TYPE_BIN_MLP */
    trt_model_header_t hdr;
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    std::vector<float> b2;
    std::vector<float> w3;
    std::vector<float> b3;

    /* Reusable activation buffers for inference */
    std::vector<float> act1;
    std::vector<float> act2;

#ifdef HAVE_CUDA
    cudaStream_t stream = nullptr;
    void        *d_input = nullptr;
    void        *d_output = nullptr;
#endif

#ifdef HAVE_TENSORRT
    std::unique_ptr<nvinfer1::IRuntime>          runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
#endif
};

#ifdef HAVE_CUDA
static void check_cuda_error(
    cudaError_t err,
    const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA Error] " << msg << ": " << cudaGetErrorString(err) << std::endl;
    }
}
#endif

/**
 * @brief Initialize an inference engine from a model file.
 */
extern "C" trt_context_t *trt_backend_create(
    const char *model_path,
    int         gpu_id)
{
    if (!model_path || strlen(model_path) == 0)
    {
        std::cerr << "[trtinfer] Error: invalid model path" << std::endl;
        return nullptr;
    }

    auto ctx = std::make_unique<trt_context>();
    strncpy(ctx->model_path, model_path, sizeof(ctx->model_path) - 1);
    ctx->gpu_id          = gpu_id;
    ctx->last_latency_us = 0.0f;
    ctx->model_type      = MODEL_TYPE_UNKNOWN;

#ifdef HAVE_CUDA
    if (gpu_id >= 0)
    {
        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > gpu_id)
        {
            cudaSetDevice(gpu_id);
            cudaStreamCreate(&ctx->stream);
        }
        else
        {
            std::cerr << "[trtinfer] Warning: GPU ID " << gpu_id << " unavailable, falling back to CPU."
                      << std::endl;
            ctx->gpu_id = -1;
        }
    }
#else
    ctx->gpu_id = -1;
#endif

    /* Check file extension */
    const char *dot = strrchr(model_path, '.');
    if (dot && strcmp(dot, ".bin") == 0)
    {
        ctx->model_type = MODEL_TYPE_BIN_MLP;

        FILE *fp = fopen(model_path, "rb");
        if (!fp)
        {
            std::cerr << "[trtinfer] Error: cannot open model file: " << model_path << std::endl;
            return nullptr;
        }

        if (fread(&ctx->hdr, sizeof(trt_model_header_t), 1, fp) != 1)
        {
            std::cerr << "[trtinfer] Error: failed reading model header" << std::endl;
            fclose(fp);
            return nullptr;
        }

        if (ctx->hdr.magic != TRT_MODEL_MAGIC)
        {
            std::cerr << "[trtinfer] Error: invalid model magic (0x" << std::hex << ctx->hdr.magic
                      << ")" << std::endl;
            fclose(fp);
            return nullptr;
        }

        /* Resize weight buffers */
        ctx->w1.resize(ctx->hdr.hidden1 * ctx->hdr.in_dim);
        ctx->b1.resize(ctx->hdr.hidden1);
        ctx->w2.resize(ctx->hdr.hidden2 * ctx->hdr.hidden1);
        ctx->b2.resize(ctx->hdr.hidden2);
        ctx->w3.resize(ctx->hdr.out_dim * ctx->hdr.hidden2);
        ctx->b3.resize(ctx->hdr.out_dim);

        if (fread(ctx->w1.data(), sizeof(float), ctx->w1.size(), fp) != ctx->w1.size() ||
            fread(ctx->b1.data(), sizeof(float), ctx->b1.size(), fp) != ctx->b1.size() ||
            fread(ctx->w2.data(), sizeof(float), ctx->w2.size(), fp) != ctx->w2.size() ||
            fread(ctx->b2.data(), sizeof(float), ctx->b2.size(), fp) != ctx->b2.size() ||
            fread(ctx->w3.data(), sizeof(float), ctx->w3.size(), fp) != ctx->w3.size() ||
            fread(ctx->b3.data(), sizeof(float), ctx->b3.size(), fp) != ctx->b3.size())
        {
            std::cerr << "[trtinfer] Error: failed reading model weight arrays" << std::endl;
            fclose(fp);
            return nullptr;
        }
        fclose(fp);

        /* Configure tensor descriptors */
        strncpy(ctx->in_desc.name, "input", sizeof(ctx->in_desc.name) - 1);
        ctx->in_desc.nb_dims      = 2;
        ctx->in_desc.dims[0]      = ctx->hdr.in_dim;
        ctx->in_desc.dims[1]      = 1;
        ctx->in_desc.num_elements = ctx->hdr.in_dim;
        ctx->in_desc.is_input     = 1;

        strncpy(ctx->out_desc.name, "output", sizeof(ctx->out_desc.name) - 1);
        ctx->out_desc.nb_dims      = 2;
        ctx->out_desc.dims[0]      = ctx->hdr.out_dim;
        ctx->out_desc.dims[1]      = 1;
        ctx->out_desc.num_elements = ctx->hdr.out_dim;
        ctx->out_desc.is_input     = 0;

        ctx->h_input.resize(ctx->in_desc.num_elements);
        ctx->h_output.resize(ctx->out_desc.num_elements);
        ctx->act1.resize(ctx->hdr.hidden1);
        ctx->act2.resize(ctx->hdr.hidden2);

#ifdef HAVE_CUDA
        if (ctx->gpu_id >= 0)
        {
            cudaMalloc(&ctx->d_input, ctx->in_desc.num_elements * sizeof(float));
            cudaMalloc(&ctx->d_output, ctx->out_desc.num_elements * sizeof(float));
        }
#endif
        std::cout << "[trtinfer] Loaded 3-layer MLP model: " << ctx->hdr.in_dim << " -> "
                  << ctx->hdr.hidden1 << " -> " << ctx->hdr.hidden2 << " -> " << ctx->hdr.out_dim
                  << " (Trained RMSE: " << ctx->hdr.final_rmse << ")" << std::endl;
    }
#ifdef HAVE_TENSORRT
    else if (dot && strcmp(dot, ".engine") == 0)
    {
        ctx->model_type = MODEL_TYPE_TRT_ENGINE;
        std::ifstream file(model_path, std::ios::binary);
        if (!file)
        {
            std::cerr << "[trtinfer] Error: cannot open engine file: " << model_path << std::endl;
            return nullptr;
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        ctx->runtime.reset(nvinfer1::createInferRuntime(gLogger));
        ctx->engine.reset(ctx->runtime->deserializeCudaEngine(engine_data.data(), size));
        if (!ctx->engine)
        {
            std::cerr << "[trtinfer] Error: failed to deserialize CUDA engine" << std::endl;
            return nullptr;
        }
        ctx->context.reset(ctx->engine->createExecutionContext());

        /* Introspect I/O tensors */
        int nb_tensors = ctx->engine->getNbIOTensors();
        for (int i = 0; i < nb_tensors; i++)
        {
            const char *tname = ctx->engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = ctx->engine->getTensorIOMode(tname);
            nvinfer1::Dims dims = ctx->engine->getTensorShape(tname);

            uint64_t count = 1;
            for (int d = 0; d < dims.nbDims; d++)
            {
                count *= (dims.d[d] > 0) ? dims.d[d] : 1;
            }

            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                strncpy(ctx->in_desc.name, tname, sizeof(ctx->in_desc.name) - 1);
                ctx->in_desc.nb_dims = dims.nbDims;
                for (int d = 0; d < dims.nbDims; d++) ctx->in_desc.dims[d] = dims.d[d];
                ctx->in_desc.num_elements = count;
                ctx->in_desc.is_input = 1;
                ctx->h_input.resize(count);
#ifdef HAVE_CUDA
                cudaMalloc(&ctx->d_input, count * sizeof(float));
                ctx->context->setTensorAddress(tname, ctx->d_input);
#endif
            }
            else
            {
                strncpy(ctx->out_desc.name, tname, sizeof(ctx->out_desc.name) - 1);
                ctx->out_desc.nb_dims = dims.nbDims;
                for (int d = 0; d < dims.nbDims; d++) ctx->out_desc.dims[d] = dims.d[d];
                ctx->out_desc.num_elements = count;
                ctx->out_desc.is_input = 0;
                ctx->h_output.resize(count);
#ifdef HAVE_CUDA
                cudaMalloc(&ctx->d_output, count * sizeof(float));
                ctx->context->setTensorAddress(tname, ctx->d_output);
#endif
            }
        }
        std::cout << "[trtinfer] Deserialized TensorRT engine: " << model_path << std::endl;
    }
    else if (dot && strcmp(dot, ".onnx") == 0)
    {
        ctx->model_type = MODEL_TYPE_ONNX;
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        const auto explicitBatch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network =
            std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        auto parser =
            std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

        if (!parser->parseFromFile(model_path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            std::cerr << "[trtinfer] Error: failed to parse ONNX model: " << model_path << std::endl;
            return nullptr;
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan)
        {
            std::cerr << "[trtinfer] Error: failed to build serialized network" << std::endl;
            return nullptr;
        }

        ctx->runtime.reset(nvinfer1::createInferRuntime(gLogger));
        ctx->engine.reset(ctx->runtime->deserializeCudaEngine(plan->data(), plan->size()));
        ctx->context.reset(ctx->engine->createExecutionContext());

        /* Introspect I/O tensors */
        int nb_tensors = ctx->engine->getNbIOTensors();
        for (int i = 0; i < nb_tensors; i++)
        {
            const char *tname = ctx->engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = ctx->engine->getTensorIOMode(tname);
            nvinfer1::Dims dims = ctx->engine->getTensorShape(tname);

            uint64_t count = 1;
            for (int d = 0; d < dims.nbDims; d++)
            {
                count *= (dims.d[d] > 0) ? dims.d[d] : 1;
            }

            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                strncpy(ctx->in_desc.name, tname, sizeof(ctx->in_desc.name) - 1);
                ctx->in_desc.nb_dims = dims.nbDims;
                for (int d = 0; d < dims.nbDims; d++) ctx->in_desc.dims[d] = dims.d[d];
                ctx->in_desc.num_elements = count;
                ctx->in_desc.is_input = 1;
                ctx->h_input.resize(count);
#ifdef HAVE_CUDA
                cudaMalloc(&ctx->d_input, count * sizeof(float));
                ctx->context->setTensorAddress(tname, ctx->d_input);
#endif
            }
            else
            {
                strncpy(ctx->out_desc.name, tname, sizeof(ctx->out_desc.name) - 1);
                ctx->out_desc.nb_dims = dims.nbDims;
                for (int d = 0; d < dims.nbDims; d++) ctx->out_desc.dims[d] = dims.d[d];
                ctx->out_desc.num_elements = count;
                ctx->out_desc.is_input = 0;
                ctx->h_output.resize(count);
#ifdef HAVE_CUDA
                cudaMalloc(&ctx->d_output, count * sizeof(float));
                ctx->context->setTensorAddress(tname, ctx->d_output);
#endif
            }
        }
        std::cout << "[trtinfer] Built and initialized TensorRT engine from ONNX: " << model_path << std::endl;
    }
#endif
    else
    {
        std::cerr << "[trtinfer] Error: unsupported model file format for: " << model_path << std::endl;
        std::cerr << "          Supported: .bin (standalone MLP)"
#ifdef HAVE_TENSORRT
                  << ", .engine (TensorRT), .onnx (ONNX model)"
#else
                  << " (Install TensorRT to enable .engine and .onnx models)"
#endif
                  << std::endl;
        return nullptr;
    }

    return ctx.release();
}

/**
 * @brief Query input tensor descriptor.
 */
extern "C" int trt_backend_get_input_desc(
    trt_context_t     *ctx,
    trt_tensor_desc_t *desc)
{
    if (!ctx || !desc)
    {
        return -1;
    }
    *desc = ctx->in_desc;
    return 0;
}

/**
 * @brief Query output tensor descriptor.
 */
extern "C" int trt_backend_get_output_desc(
    trt_context_t     *ctx,
    trt_tensor_desc_t *desc)
{
    if (!ctx || !desc)
    {
        return -1;
    }
    *desc = ctx->out_desc;
    return 0;
}

/**
 * @brief Copy input data from host buffer to inference engine.
 */
extern "C" errno_t trt_backend_set_input(
    trt_context_t *ctx,
    const float   *h_input,
    size_t         count)
{
    if (!ctx || !h_input || count != ctx->in_desc.num_elements)
    {
        return EINVAL;
    }

    memcpy(ctx->h_input.data(), h_input, count * sizeof(float));

#ifdef HAVE_CUDA
    if (ctx->gpu_id >= 0 && ctx->d_input)
    {
        cudaMemcpyAsync(ctx->d_input, h_input, count * sizeof(float),
                        cudaMemcpyHostToDevice, ctx->stream);
    }
#endif
    return 0;
}

/**
 * @brief Execute model inference.
 */
extern "C" errno_t trt_backend_execute(
    trt_context_t *ctx)
{
    if (!ctx)
    {
        return EINVAL;
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    if (ctx->model_type == MODEL_TYPE_BIN_MLP)
    {
        const float *px = ctx->h_input.data();
        float       *h1 = ctx->act1.data();
        float       *h2 = ctx->act2.data();
        float       *py = ctx->h_output.data();

        int in_dim  = (int) ctx->hdr.in_dim;
        int h1_dim  = (int) ctx->hdr.hidden1;
        int h2_dim  = (int) ctx->hdr.hidden2;
        int out_dim = (int) ctx->hdr.out_dim;

        /* Layer 1: h1 = ReLU(W1 * x + b1) */
        for (int h = 0; h < h1_dim; h++)
        {
            float sum = ctx->b1[h];
            for (int k = 0; k < in_dim; k++)
            {
                sum += ctx->w1[h * in_dim + k] * px[k];
            }
            h1[h] = (sum > 0.0f) ? sum : 0.0f;
        }

        /* Layer 2: h2 = ReLU(W2 * h1 + b2) */
        for (int h = 0; h < h2_dim; h++)
        {
            float sum = ctx->b2[h];
            for (int k = 0; k < h1_dim; k++)
            {
                sum += ctx->w2[h * h1_dim + k] * h1[k];
            }
            h2[h] = (sum > 0.0f) ? sum : 0.0f;
        }

        /* Layer 3: py = W3 * h2 + b3 */
        for (int o = 0; o < out_dim; o++)
        {
            float sum = ctx->b3[o];
            for (int k = 0; k < h2_dim; k++)
            {
                sum += ctx->w3[o * h2_dim + k] * h2[k];
            }
            py[o] = sum;
        }
    }
#if defined(HAVE_TENSORRT) && defined(HAVE_CUDA)
    else if (ctx->model_type == MODEL_TYPE_TRT_ENGINE || ctx->model_type == MODEL_TYPE_ONNX)
    {
        ctx->context->enqueueV3(ctx->stream);
        cudaStreamSynchronize(ctx->stream);
        cudaMemcpyAsync(ctx->h_output.data(), ctx->d_output,
                        ctx->out_desc.num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    ctx->last_latency_us = (float) ((ts_end.tv_sec - ts_start.tv_sec) * 1e6 +
                                    (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-3);

    return 0;
}

/**
 * @brief Retrieve output data from inference engine into host buffer.
 */
extern "C" errno_t trt_backend_get_output(
    trt_context_t *ctx,
    float         *h_output,
    size_t         count)
{
    if (!ctx || !h_output || count != ctx->out_desc.num_elements)
    {
        return EINVAL;
    }

    memcpy(h_output, ctx->h_output.data(), count * sizeof(float));
    return 0;
}

/**
 * @brief Get latest execution latency in microseconds.
 */
extern "C" float trt_backend_get_latency_us(
    const trt_context_t *ctx)
{
    return ctx ? ctx->last_latency_us : 0.0f;
}

/**
 * @brief Destroy inference context and free all allocated memory.
 */
extern "C" void trt_backend_destroy(
    trt_context_t *ctx)
{
    if (!ctx)
    {
        return;
    }

#ifdef HAVE_CUDA
    if (ctx->d_input)  cudaFree(ctx->d_input);
    if (ctx->d_output) cudaFree(ctx->d_output);
    if (ctx->stream)   cudaStreamDestroy(ctx->stream);
#endif

    delete ctx;
}
