// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_exec.h
 * @brief   Declarations for trtinfer neural network inference compute unit
 */

#ifndef TRTINFER_EXEC_H
#define TRTINFER_EXEC_H

#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
/**
 * @brief Register trtinfer CLI command.
 *
 * @return RETURN_SUCCESS on success.
 */
errno_t CLIADDCMD_trtinfer__trtinfer(void);
#endif

#ifdef __cplusplus
}
#endif

#endif // TRTINFER_EXEC_H
