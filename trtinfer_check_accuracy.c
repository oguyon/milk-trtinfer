// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_check_accuracy.c
 * @brief   Standalone C validator for trtinfer 3D dataset and inference accuracy
 *
 * Attaches directly to ImageStreamIO shared memory streams and verifies:
 * 1. That generated dataset points match analytical 3D coupled optical swirl formula (< 1e-5).
 * 2. Optionally, that an inferred stream (e.g. pts_out) accurately reproduces the ground truth (< 0.05 RMSE).
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "trtinfer_datagen.h"

#define TOLERANCE_DATA_MAX_ERR 1e-5
#define TOLERANCE_INFER_RMSE   0.05

int main(
    int    argc,
    char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr,
                "Usage: %s <in_stream_name> <truth_stream_name> [binary_file] [inferred_stream_name]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_name     = argv[1];
    const char *truth_name  = argv[2];
    const char *bin_path    = (argc >= 4 && strlen(argv[3]) > 0 && strcmp(argv[3], "none") != 0)
                              ? argv[3] : NULL;
    const char *infer_name  = (argc >= 5 && strlen(argv[4]) > 0) ? argv[4] : NULL;

    /* 1. Connect to input stream */
    IMAGE im_in;
    memset(&im_in, 0, sizeof(IMAGE));
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &im_in) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: cannot connect to input stream '%s'\n", in_name);
        return EXIT_FAILURE;
    }

    /* 2. Connect to truth stream */
    IMAGE im_truth;
    memset(&im_truth, 0, sizeof(IMAGE));
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(truth_name, &im_truth) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: cannot connect to truth stream '%s'\n", truth_name);
        ImageStreamIO_closeIm(&im_in);
        return EXIT_FAILURE;
    }

    /* 3. Validate metadata */
    if (im_in.md->datatype != _DATATYPE_FLOAT || im_truth.md->datatype != _DATATYPE_FLOAT)
    {
        fprintf(stderr, "Error: stream datatype must be _DATATYPE_FLOAT\n");
        ImageStreamIO_closeIm(&im_in);
        ImageStreamIO_closeIm(&im_truth);
        return EXIT_FAILURE;
    }

    if (im_in.md->size[0] != 3 || im_truth.md->size[0] != 3)
    {
        fprintf(stderr, "Error: expected size[0] == 3 (got in=%u, truth=%u)\n",
                im_in.md->size[0], im_truth.md->size[0]);
        ImageStreamIO_closeIm(&im_in);
        ImageStreamIO_closeIm(&im_truth);
        return EXIT_FAILURE;
    }

    if (im_in.md->size[1] != im_truth.md->size[1])
    {
        fprintf(stderr, "Error: sample count mismatch (in=%u, truth=%u)\n",
                im_in.md->size[1], im_truth.md->size[1]);
        ImageStreamIO_closeIm(&im_in);
        ImageStreamIO_closeIm(&im_truth);
        return EXIT_FAILURE;
    }

    uint32_t nsamples = im_in.md->size[1];
    const float *p_in    = im_in.array.F;
    const float *p_truth = im_truth.array.F;

    /* 4. Numerically verify dataset against analytical formula */
    double max_err    = 0.0;
    double sum_sq_err = 0.0;

    for (uint32_t i = 0; i < nsamples; i++)
    {
        float x1 = p_in[i * 3 + 0];
        float x2 = p_in[i * 3 + 1];
        float x3 = p_in[i * 3 + 2];

        float y1_ref = 0.0f;
        float y2_ref = 0.0f;
        float y3_ref = 0.0f;
        trtinfer_map_3d_point(x1, x2, x3, &y1_ref, &y2_ref, &y3_ref);

        float y1_act = p_truth[i * 3 + 0];
        float y2_act = p_truth[i * 3 + 1];
        float y3_act = p_truth[i * 3 + 2];

        double d1 = fabs((double) y1_act - (double) y1_ref);
        double d2 = fabs((double) y2_act - (double) y2_ref);
        double d3 = fabs((double) y3_act - (double) y3_ref);

        if (d1 > max_err)
        {
            max_err = d1;
        }
        if (d2 > max_err)
        {
            max_err = d2;
        }
        if (d3 > max_err)
        {
            max_err = d3;
        }

        sum_sq_err += d1 * d1 + d2 * d2 + d3 * d3;
    }

    double rmse = sqrt(sum_sq_err / (3.0 * (double) nsamples));

    /* 5. Optional check against exported binary file */
    if (bin_path != NULL)
    {
        FILE *fp = fopen(bin_path, "rb");
        if (fp == NULL)
        {
            fprintf(stderr, "Error: cannot open binary file '%s'\n", bin_path);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        uint32_t hdr[2];
        if (fread(hdr, sizeof(uint32_t), 2, fp) != 2 || hdr[0] != 3 || hdr[1] != nsamples)
        {
            fprintf(stderr, "Error: binary header mismatch\n");
            fclose(fp);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        float *b_in    = (float *) malloc(sizeof(float) * 3 * nsamples);
        float *b_truth = (float *) malloc(sizeof(float) * 3 * nsamples);

        if (b_in == NULL || b_truth == NULL)
        {
            fprintf(stderr, "Error: memory allocation failure during binary check\n");
            free(b_in);
            free(b_truth);
            fclose(fp);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        size_t r1 = fread(b_in, sizeof(float), 3 * nsamples, fp);
        size_t r2 = fread(b_truth, sizeof(float), 3 * nsamples, fp);
        fclose(fp);

        if (r1 != 3 * nsamples || r2 != 3 * nsamples)
        {
            fprintf(stderr, "Error: incomplete binary file read\n");
            free(b_in);
            free(b_truth);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        if (memcmp(p_in, b_in, sizeof(float) * 3 * nsamples) != 0 ||
            memcmp(p_truth, b_truth, sizeof(float) * 3 * nsamples) != 0)
        {
            fprintf(stderr, "Error: binary file content differs from shared memory stream\n");
            free(b_in);
            free(b_truth);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        free(b_in);
        free(b_truth);
    }

    printf("Dataset validation:\n");
    printf("  Samples verified:                    %u\n", nsamples);
    printf("  Max discrepancy vs analytical model: %.3e\n", max_err);
    printf("  RMSE vs analytical model:            %.3e\n", rmse);

    if (max_err > TOLERANCE_DATA_MAX_ERR)
    {
        fprintf(stderr, "FAILED: max discrepancy %.3e exceeds tolerance %.3e\n",
                max_err, TOLERANCE_DATA_MAX_ERR);
        ImageStreamIO_closeIm(&im_in);
        ImageStreamIO_closeIm(&im_truth);
        return EXIT_FAILURE;
    }

    /* 6. Optional verification of inferred output stream */
    if (infer_name != NULL)
    {
        IMAGE im_infer;
        memset(&im_infer, 0, sizeof(IMAGE));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(infer_name, &im_infer) != IMAGESTREAMIO_SUCCESS)
        {
            fprintf(stderr, "Error: cannot connect to inferred stream '%s'\n", infer_name);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        if (im_infer.md->datatype != _DATATYPE_FLOAT ||
            im_infer.md->size[0] != 3 || im_infer.md->size[1] != nsamples)
        {
            fprintf(stderr, "Error: inferred stream dimensions/type mismatch (%ux%u)\n",
                    im_infer.md->size[0], im_infer.md->size[1]);
            ImageStreamIO_closeIm(&im_infer);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }

        const float *p_infer = im_infer.array.F;
        double infer_sum_sq = 0.0;
        double infer_max_err = 0.0;

        for (uint32_t i = 0; i < nsamples; i++)
        {
            for (int k = 0; k < 3; k++)
            {
                double diff = fabs((double) p_infer[i * 3 + k] - (double) p_truth[i * 3 + k]);
                if (diff > infer_max_err)
                {
                    infer_max_err = diff;
                }
                infer_sum_sq += diff * diff;
            }
        }

        double infer_rmse = sqrt(infer_sum_sq / (3.0 * (double) nsamples));
        ImageStreamIO_closeIm(&im_infer);

        printf("Inference accuracy validation ('%s' vs '%s'):\n", infer_name, truth_name);
        printf("  Inference Max Absolute Error:        %.4f\n", infer_max_err);
        printf("  Inference RMSE:                      %.4f (Pass threshold: < %.2f)\n",
               infer_rmse, TOLERANCE_INFER_RMSE);

        if (infer_rmse > TOLERANCE_INFER_RMSE)
        {
            fprintf(stderr, "FAILED: inference RMSE %.4f exceeds threshold %.2f\n",
                    infer_rmse, TOLERANCE_INFER_RMSE);
            ImageStreamIO_closeIm(&im_in);
            ImageStreamIO_closeIm(&im_truth);
            return EXIT_FAILURE;
        }
        printf("  Inference Accuracy Check:            [PASS]\n");
    }

    ImageStreamIO_closeIm(&im_in);
    ImageStreamIO_closeIm(&im_truth);

    return EXIT_SUCCESS;
}
