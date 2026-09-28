// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_model.h
 * @brief   Data structures and header definitions for standalone neural network models
 */

#ifndef TRTINFER_MODEL_H
#define TRTINFER_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRT_MODEL_MAGIC 0x4D4C5031U /* "MLP1" in ASCII hex */

/**
 * @brief Serialized model header structure for multi-layer perceptron.
 */
typedef struct
{
    uint32_t magic;      /**< Magic identifier: TRT_MODEL_MAGIC */
    uint32_t num_layers; /**< Total layers (3 for input->h1->h2->output) */
    uint32_t in_dim;     /**< Input dimension (e.g. 3) */
    uint32_t hidden1;    /**< Hidden layer 1 dimension (e.g. 64) */
    uint32_t hidden2;    /**< Hidden layer 2 dimension (e.g. 64) */
    uint32_t out_dim;    /**< Output dimension (e.g. 3) */
    float    final_rmse; /**< Training root mean square error */
    uint32_t reserved;   /**< Alignment padding */
} trt_model_header_t;

#ifdef __cplusplus
}
#endif

#endif // TRTINFER_MODEL_H
