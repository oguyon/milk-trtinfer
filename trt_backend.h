// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trt_backend.h
 * @brief   Opaque C API for TensorRT and neural network inference runtime
 */

#ifndef TRT_BACKEND_H
#define TRT_BACKEND_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#ifndef _ERRNO_T_DEFINED
#define _ERRNO_T_DEFINED
typedef int errno_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to an inference engine context.
 */
typedef struct trt_context trt_context_t;

/**
 * @brief Tensor metadata descriptor.
 */
typedef struct {
    char     name[64];
    int      nb_dims;
    int64_t  dims[8];
    uint64_t num_elements;
    int      is_input;
} trt_tensor_desc_t;

/**
 * @brief Initialize an inference engine from a model file (.engine, .onnx, or .bin).
 *
 * @param[in] model_path  Path to serialized model or engine
 * @param[in] gpu_id      GPU device ID (-1 for CPU execution)
 * @return Opaque context pointer on success, NULL on failure
 */
trt_context_t *trt_backend_create(
    const char *model_path,
    int         gpu_id);

/**
 * @brief Query input tensor descriptor.
 *
 * @param[in]  ctx   Inference context
 * @param[out] desc  Tensor descriptor to populate
 * @return 0 on success, -1 on failure
 */
int trt_backend_get_input_desc(
    trt_context_t     *ctx,
    trt_tensor_desc_t *desc);

/**
 * @brief Query output tensor descriptor.
 *
 * @param[in]  ctx   Inference context
 * @param[out] desc  Tensor descriptor to populate
 * @return 0 on success, -1 on failure
 */
int trt_backend_get_output_desc(
    trt_context_t     *ctx,
    trt_tensor_desc_t *desc);

/**
 * @brief Copy input data from host buffer to inference engine.
 *
 * @param[in] ctx      Inference context
 * @param[in] h_input  Pointer to input float array
 * @param[in] count    Number of float elements
 * @return 0 on success, non-zero error code on failure
 */
errno_t trt_backend_set_input(
    trt_context_t *ctx,
    const float   *h_input,
    size_t         count);

/**
 * @brief Execute model inference.
 *
 * @param[in] ctx  Inference context
 * @return 0 on success, non-zero error code on failure
 */
errno_t trt_backend_execute(
    trt_context_t *ctx);

/**
 * @brief Retrieve output data from inference engine into host buffer.
 *
 * @param[in]  ctx      Inference context
 * @param[out] h_output Pointer to destination float array
 * @param[in]  count    Number of float elements
 * @return 0 on success, non-zero error code on failure
 */
errno_t trt_backend_get_output(
    trt_context_t *ctx,
    float         *h_output,
    size_t         count);

/**
 * @brief Get latest execution latency in microseconds.
 *
 * @param[in] ctx  Inference context
 * @return Execution latency in microseconds
 */
float trt_backend_get_latency_us(
    const trt_context_t *ctx);

/**
 * @brief Destroy inference context and free all allocated memory.
 *
 * @param[in] ctx  Inference context
 */
void trt_backend_destroy(
    trt_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // TRT_BACKEND_H
