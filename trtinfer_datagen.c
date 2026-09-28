// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_datagen.c
 * @brief   Non-linear 3D dataset and trajectory generator for TensorRT inference
 *
 * Implements a non-linear 3D -> 3D coordinate mapping based on coupled optical
 * swirl warping. Generates training datasets or streaming 3D trajectories into
 * native ImageStreamIO shared memory streams.
 */

#include <math.h>
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

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "fps.h"
#include "trtinfer_datagen.h"

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "trtdatagen",
    .cmdkey      = "trtinfer_datagen",
    .description = "Generate non-linear 3D dataset in ImageStreamIO streams"
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     param_out_instream[FUNCTION_PARAMETER_STRMAXLEN]    = "pts_in";
static char     param_out_truthstream[FUNCTION_PARAMETER_STRMAXLEN] = "pts_truth";
static uint32_t param_nsamples                                      = 10000;
static int32_t  param_mode                                          = 0;
static char     param_save_fits[FUNCTION_PARAMETER_STRMAXLEN]       = "data_3d.fits";
static float    param_stream_rate_hz                                = 0.0f;

/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".out_instream", param_out_instream, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_OUTPUT_STREAM,       \
      "Input points stream name (pts_in)")                                                         \
    X(".out_truthstream", param_out_truthstream, FPTYPE_STREAMNAME, 1,                             \
      FPFLAG_DEFAULT_OUTPUT_STREAM, "Ground truth stream name (pts_truth)")                        \
    X(".nsamples", &param_nsamples, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT,                        \
      "Number of 3D samples")                                                                      \
    X(".save_fits", param_save_fits, FPTYPE_FILENAME, 1, FPFLAG_DEFAULT_INPUT,                     \
      "Optional FITS or binary output file path")                                                  \
    X(".mode", &param_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                                 \
      "Sampling mode (0: random, 1: grid, 2: spiral)")                                             \
    X(".stream_rate_hz", &param_stream_rate_hz, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,           \
      "Streaming rate in Hz (0: single batch generation)")

/* ================================================================
 * 4.  COMPUTATION LOGIC & MAPPING
 * ============================================================= */

/**
 * @brief Maps a 3D input point to a 3D output point via coupled optical swirl.
 */
errno_t trtinfer_map_3d_point(
    float  x1,
    float  x2,
    float  x3,
    float *y1,
    float *y2,
    float *y3)
{
    if (y1 == NULL || y2 == NULL || y3 == NULL)
    {
        return RETURN_FAILURE;
    }

    /* Coupled optical swirl parameters */
    const float omega = 0.5f * PI;
    const float alpha = 0.25f;
    const float beta  = 1.20f;
    const float gamma = 0.35f;
    const float eps   = 1e-6f;

    float r12   = sqrtf(x1 * x1 + x2 * x2 + eps);
    float theta = omega * r12;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float x3_sq = x3 * x3;

    *y1 = x1 * cos_t - x2 * sin_t + alpha * x3_sq;
    *y2 = x1 * sin_t + x2 * cos_t - alpha * x3_sq;
    *y3 = tanhf(beta * x3) + gamma * sinf(PI * x1 * x2);

    return RETURN_SUCCESS;
}

/**
 * @brief Custom configuration check callback.
 */
static MILK_COLD errno_t customCONFcheck(void)
{
    if (param_nsamples == 0)
    {
        param_nsamples = 1000;
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

    int is_streaming = (param_stream_rate_hz > 0.0f);
    uint32_t ysize   = is_streaming ? 1 : param_nsamples;

    /* Connect or create 2D float ImageStreamIO streams */
    IMGID img_in = stream_connect_create_2Df32(param_out_instream, 3, ysize);
    if (img_in.ID == -1)
    {
        fprintf(stderr, "Error: failed to create input stream '%s'\n", param_out_instream);
        return RETURN_FAILURE;
    }

    IMGID img_truth = stream_connect_create_2Df32(param_out_truthstream, 3, ysize);
    if (img_truth.ID == -1)
    {
        fprintf(stderr, "Error: failed to create truth stream '%s'\n", param_out_truthstream);
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    if (!is_streaming)
    {
        /* BATCH MODE: generate all N samples once */
        float *restrict p_in    = img_in.im->array.F;
        float *restrict p_truth = img_truth.im->array.F;

        srand((unsigned int) time(NULL));

        if (param_mode == 1)
        {
            /* Regular 3D grid */
            uint32_t k = (uint32_t) cbrtf((float) param_nsamples);
            if (k < 2)
            {
                k = 2;
            }
            float step = 2.0f / (float) (k - 1);
            uint32_t idx = 0;

            for (uint32_t iz = 0; iz < k && idx < param_nsamples; iz++)
            {
                float z = -1.0f + (float) iz * step;
                for (uint32_t iy = 0; iy < k && idx < param_nsamples; iy++)
                {
                    float y = -1.0f + (float) iy * step;
                    for (uint32_t ix = 0; ix < k && idx < param_nsamples; ix++)
                    {
                        float x = -1.0f + (float) ix * step;
                        p_in[idx * 3 + 0] = x;
                        p_in[idx * 3 + 1] = y;
                        p_in[idx * 3 + 2] = z;
                        trtinfer_map_3d_point(
                            x, y, z,
                            &p_truth[idx * 3 + 0],
                            &p_truth[idx * 3 + 1],
                            &p_truth[idx * 3 + 2]);
                        idx++;
                    }
                }
            }
            /* Fill any remaining samples if nsamples > k^3 */
            for (; idx < param_nsamples; idx++)
            {
                float x = -1.0f + 2.0f * ((float) rand() / (float) RAND_MAX);
                float y = -1.0f + 2.0f * ((float) rand() / (float) RAND_MAX);
                float z = -1.0f + 2.0f * ((float) rand() / (float) RAND_MAX);
                p_in[idx * 3 + 0] = x;
                p_in[idx * 3 + 1] = y;
                p_in[idx * 3 + 2] = z;
                trtinfer_map_3d_point(
                    x, y, z,
                    &p_truth[idx * 3 + 0],
                    &p_truth[idx * 3 + 1],
                    &p_truth[idx * 3 + 2]);
            }
        }
        else if (param_mode == 2)
        {
            /* 3D Spiral / Knot parametric trajectory */
            for (uint32_t i = 0; i < param_nsamples; i++)
            {
                float t = (2.0f * PI * (float) i) / (float) param_nsamples;
                float x = sinf(t) * cosf(3.0f * t);
                float y = sinf(t) * sinf(3.0f * t);
                float z = cosf(t);
                p_in[i * 3 + 0] = x;
                p_in[i * 3 + 1] = y;
                p_in[i * 3 + 2] = z;
                trtinfer_map_3d_point(
                    x, y, z,
                    &p_truth[i * 3 + 0],
                    &p_truth[i * 3 + 1],
                    &p_truth[i * 3 + 2]);
            }
        }
        else
        {
            /* Random Uniform sampling in [-1, 1]^3 */
            for (uint32_t i = 0; i < param_nsamples; i++)
            {
                float x = -1.0f + 2.0f * ((float) rand() / (float) RAND_MAX);
                float y = -1.0f + 2.0f * ((float) rand() / (float) RAND_MAX);
                float z = -1.0f + 2.0f * ((float) rand() / (float) RAND_MAX);
                p_in[i * 3 + 0] = x;
                p_in[i * 3 + 1] = y;
                p_in[i * 3 + 2] = z;
                trtinfer_map_3d_point(
                    x, y, z,
                    &p_truth[i * 3 + 0],
                    &p_truth[i * 3 + 1],
                    &p_truth[i * 3 + 2]);
            }
        }

        /* Post updates to ImageStreamIO streams */
        img_in.im->md->write = 1;
        processinfo_update_output_stream(processinfo, img_in.im, NULL);

        img_truth.im->md->write = 1;
        processinfo_update_output_stream(processinfo, img_truth.im, NULL);

        printf("[trtinfer_datagen] Generated %u samples into '%s' and '%s'\n",
               param_nsamples, param_out_instream, param_out_truthstream);

        /* Save to FITS if requested and available */
        if (param_save_fits[0] != '\0')
        {
            char fname_in[FUNCTION_PARAMETER_STRMAXLEN + 16];
            char fname_truth[FUNCTION_PARAMETER_STRMAXLEN + 16];
            snprintf(fname_in, sizeof(fname_in), "%s_in.fits", param_save_fits);
            snprintf(fname_truth, sizeof(fname_truth), "%s_truth.fits", param_save_fits);

            save_fits(param_out_instream, fname_in);
            save_fits(param_out_truthstream, fname_truth);
            printf("[trtinfer_datagen] Saved FITS files: %s, %s\n",
                   fname_in, fname_truth);

            /* Also save binary file for easy ingestion */
            char fname_bin[FUNCTION_PARAMETER_STRMAXLEN + 16];
            snprintf(fname_bin, sizeof(fname_bin), "%s.bin", param_save_fits);
            FILE *fp = fopen(fname_bin, "wb");
            if (fp != NULL)
            {
                uint32_t header[2] = { 3, param_nsamples };
                fwrite(header, sizeof(uint32_t), 2, fp);
                fwrite(p_in, sizeof(float), 3 * param_nsamples, fp);
                fwrite(p_truth, sizeof(float), 3 * param_nsamples, fp);
                fclose(fp);
                printf("[trtinfer_datagen] Saved binary file: %s\n", fname_bin);
            }
        }
    }
    else
    {
        /* STREAMING MODE: continuous trajectory generation at specified rate */
        useconds_t sleep_us = (useconds_t) (1e6f / param_stream_rate_hz);
        uint64_t frame_idx = 0;

        INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
        {
            float t = (float) frame_idx * 0.01f;
            float x = sinf(t) * cosf(3.0f * t);
            float y = sinf(t) * sinf(3.0f * t);
            float z = cosf(t);

            img_in.im->array.F[0] = x;
            img_in.im->array.F[1] = y;
            img_in.im->array.F[2] = z;

            trtinfer_map_3d_point(
                x, y, z,
                &img_truth.im->array.F[0],
                &img_truth.im->array.F[1],
                &img_truth.im->array.F[2]);

            img_in.im->md->write = 1;
            processinfo_update_output_stream(processinfo, img_in.im, NULL);

            img_truth.im->md->write = 1;
            processinfo_update_output_stream(processinfo, img_truth.im, NULL);

            frame_idx++;
            usleep(sleep_us);
        }
        INSERT_STD_PROCINFO_COMPUTEFUNC_END
    }

    imgid_free(&img_in);
    imgid_free(&img_truth);

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

errno_t CLIADDCMD_trtinfer__trtinfer_datagen(void)
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
