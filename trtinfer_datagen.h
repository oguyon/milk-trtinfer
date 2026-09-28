// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_datagen.h
 * @brief   Declarations for non-linear 3D dataset generator
 */

#ifndef TRTINFER_DATAGEN_H
#define TRTINFER_DATAGEN_H

#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maps a 3D input point to a 3D output point via coupled optical swirl warping.
 *
 * Implements the non-linear transformation:
 *   r12 = sqrt(x1^2 + x2^2 + eps)
 *   theta = (pi / 2) * r12
 *   y1 = x1 * cos(theta) - x2 * sin(theta) + 0.25 * x3^2
 *   y2 = x1 * sin(theta) + x2 * cos(theta) - 0.25 * x3^2
 *   y3 = tanh(1.2 * x3) + 0.35 * sin(pi * x1 * x2)
 *
 * @param[in]  x1  Input x coordinate in [-1, 1]
 * @param[in]  x2  Input y coordinate in [-1, 1]
 * @param[in]  x3  Input z coordinate in [-1, 1]
 * @param[out] y1  Pointer to output y1 coordinate
 * @param[out] y2  Pointer to output y2 coordinate
 * @param[out] y3  Pointer to output y3 coordinate
 * @return RETURN_SUCCESS on success, RETURN_FAILURE on NULL pointer
 */
errno_t trtinfer_map_3d_point(
    float  x1,
    float  x2,
    float  x3,
    float *y1,
    float *y2,
    float *y3);

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
/**
 * @brief Register trtinfer_datagen CLI command.
 *
 * @return RETURN_SUCCESS on success.
 */
errno_t CLIADDCMD_trtinfer__trtinfer_datagen(void);
#endif

#ifdef __cplusplus
}
#endif

#endif // TRTINFER_DATAGEN_H
