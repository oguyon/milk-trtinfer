// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_exec.c
 * @brief   Real-time neural network inference compute unit on ImageStreamIO
 *
 * Runs TensorRT, ONNX, or standalone MLP models on ImageStreamIO shared memory streams.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "fps.h"
#include "trt_backend.h"
#include "trtinfer_exec.h"

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "trtinfer",
    .cmdkey      = "trtinfer",
    .description = "Real-time neural network inference compute unit on ImageStreamIO"
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     param_in_stream[FUNCTION_PARAMETER_STRMAXLEN]  = "pts_in";
static char     param_out_stream[FUNCTION_PARAMETER_STRMAXLEN] = "pts_out";
static char     param_model_path[FUNCTION_PARAMETER_STRMAXLEN] = "model_3d.bin";
static int32_t  param_gpu_id                                   = 0;
static int32_t  param_sync_mode                                = 0;
static float    param_rate_hz                                  = 1000.0f;
static float    param_latency_us                               = 0.0f;
static uint64_t param_sample_count                             = 0;

/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".in_stream", param_in_stream, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM,          \
      "Input stream name (e.g. pts_in)")                                                           \
    X(".out_stream", param_out_stream, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_OUTPUT_STREAM,         \
      "Output stream name (e.g. pts_out)")                                                         \
    X(".model_path", param_model_path, FPTYPE_FILENAME, 1, FPFLAG_DEFAULT_INPUT,                   \
      "Path to model file (.bin, .engine, or .onnx)")                                              \
    X(".gpu_id", &param_gpu_id, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                             \
      "GPU device ID (-1: CPU, >=0: GPU)")                                                         \
    X(".sync_mode", &param_sync_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Sync mode (0: trigger on input stream semaphore, 1: timer)")                                \
    X(".rate_hz", &param_rate_hz, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                         \
      "Timer rate in Hz when sync_mode=1")                                                         \
    X(".latency_us", &param_latency_us, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,                  \
      "Last inference latency in microseconds")                                                    \
    X(".sample_count", &param_sample_count, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,               \
      "Total inferred sample count")

/* ================================================================
 * 4.  COMPUTATION LOGIC & CONFIG CHECK
 * ============================================================= */

static MILK_COLD errno_t customCONFcheck(void)
{
    if (param_rate_hz <= 0.0f)
    {
        param_rate_hz = 1000.0f;
    }

    return RETURN_SUCCESS;
}

/* ================================================================
 * 5.  CLI BINDINGS & DEFINITIONS
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)

/* ================================================================
 * 6.  PROCESSINFO COMPUTE FUNCTION
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function(void)
{
    DEBUG_TRACE_FSTART();

    /* 1. Initialize Inference Backend */
    trt_context_t *ctx = trt_backend_create(param_model_path, param_gpu_id);
    if (ctx == NULL)
    {
        fprintf(stderr,
                "[trtinfer] Error: cannot initialize inference engine with model '%s'\n",
                param_model_path);
        return RETURN_FAILURE;
    }

    trt_tensor_desc_t in_desc, out_desc;
    trt_backend_get_input_desc(ctx, &in_desc);
    trt_backend_get_output_desc(ctx, &out_desc);

    /* 2. Connect to Input Stream */
    IMGID img_in = imgid_make_from_name(param_in_stream);
    resolveIMGID(&img_in, ERRMODE_WARN, dcimg, dcnimg);
    if (img_in.ID == -1)
    {
        read_sharedmem_image(param_in_stream, dcimg, dcnimg);
        resolveIMGID(&img_in, ERRMODE_WARN, dcimg, dcnimg);
    }

    if (img_in.ID == -1)
    {
        fprintf(stderr, "[trtinfer] Error: cannot resolve input stream '%s'\n", param_in_stream);
        trt_backend_destroy(ctx);
        return RETURN_FAILURE;
    }

    /* 3. Determine Batching or Streaming Dimensions */
    uint32_t in_dim = (uint32_t) in_desc.num_elements;
    uint32_t out_dim = (uint32_t) out_desc.num_elements;
    uint32_t nsamples = 1;

    if (img_in.md->naxis == 2 && img_in.md->size[0] == in_dim)
    {
        nsamples = img_in.md->size[1];
    }

    /* Connect or create output stream */
    IMGID img_out = stream_connect_create_2Df32(param_out_stream, out_dim, nsamples);
    if (img_out.ID == -1)
    {
        fprintf(stderr, "[trtinfer] Error: failed to create output stream '%s'\n", param_out_stream);
        imgid_free(&img_in);
        trt_backend_destroy(ctx);
        return RETURN_FAILURE;
    }

    printf("[trtinfer] Input: '%s' [%u x %u], Output: '%s' [%u x %u], Model: '%s'\n",
           param_in_stream, in_dim, nsamples,
           param_out_stream, out_dim, nsamples,
           param_model_path);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    param_sample_count = 0;
    useconds_t sleep_us = (param_rate_hz > 0.0f) ? (useconds_t) (1e6f / param_rate_hz) : 1000;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        const float *p_in  = img_in.im->array.F;
        float       *p_out = img_out.im->array.F;

        img_out.im->md->write = 1;

        if (nsamples == 1)
        {
            /* Single sample inference */
            trt_backend_set_input(ctx, p_in, in_dim);
            trt_backend_execute(ctx);
            trt_backend_get_output(ctx, p_out, out_dim);
            param_sample_count++;
        }
        else
        {
            /* Multi-sample batch inference */
            for (uint32_t i = 0; i < nsamples; i++)
            {
                trt_backend_set_input(ctx, &p_in[i * in_dim], in_dim);
                trt_backend_execute(ctx);
                trt_backend_get_output(ctx, &p_out[i * out_dim], out_dim);
                param_sample_count++;
            }
        }

        param_latency_us = trt_backend_get_latency_us(ctx);

        processinfo_update_output_stream(processinfo, img_out.im, img_in.im);

        if (param_sync_mode == 1)
        {
            usleep(sleep_us);
        }
        else if (nsamples > 1)
        {
            /* For a batch array run in batch trigger, execute once and exit loop */
            break;
        }
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&img_in);
    imgid_free(&img_out);
    trt_backend_destroy(ctx);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  CLI REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

errno_t CLIADDCMD_trtinfer__trtinfer(void)
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

/* ================================================================
 * 8.  STANDALONE EXECUTABLE MAIN
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info, FPS_PARAMS, compute_function, customCONFcheck)
#endif
