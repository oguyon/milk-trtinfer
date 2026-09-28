// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer.c
 * @brief   Main entry point for trtinfer plugin module
 */

#define MODULE_SHORTNAME_DEFAULT "trtinfer"
#define MODULE_DESCRIPTION "TensorRT and neural network inference for milk"

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif

#include "trtinfer.h"

#ifndef MILK_NO_CLI
static errno_t init_module_CLI(void)
{
    CLIADDCMD_trtinfer__trtinfer_datagen();
    CLIADDCMD_trtinfer__trtinfer();

    return RETURN_SUCCESS;
}

MILK_MODULE(trtinfer, init_module_CLI, NULL);
#endif /* MILK_NO_CLI */
