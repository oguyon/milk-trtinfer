// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    trtinfer_train_mlp.c
 * @brief   Standalone C trainer for 3D multi-layer perceptron neural network
 *
 * Trains a 3-layer MLP (3 -> 64 -> 64 -> 3) directly from ImageStreamIO shared memory
 * streams using mini-batch Adam optimization in pure C without Python or PyTorch.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "trtinfer_model.h"

#define DEFAULT_EPOCHS     500
#define DEFAULT_BATCH_SIZE 64
#define DEFAULT_LR         0.005f
#define BETA1              0.9f
#define BETA2              0.999f
#define EPSILON            1e-8f

#define DIM_IN 3
#define DIM_H1 64
#define DIM_H2 64
#define DIM_OUT 3

/**
 * @brief Fast PRNG (xorshift32) for deterministic, fast random numbers.
 */
static inline uint32_t xorshift32(
    uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * @brief Generate standard normal Gaussian variate via Box-Muller transform.
 */
static float rand_normal(
    uint32_t *state)
{
    float u1 = ((float) (xorshift32(state) & 0x7FFFFFFF)) / (float) 0x7FFFFFFF;
    float u2 = ((float) (xorshift32(state) & 0x7FFFFFFF)) / (float) 0x7FFFFFFF;
    if (u1 < 1e-7f)
    {
        u1 = 1e-7f;
    }
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float) M_PI * u2);
}

/**
 * @brief Shuffle array of indices using Fisher-Yates algorithm.
 */
static void shuffle_indices(
    uint32_t *indices,
    size_t    n,
    uint32_t *rng)
{
    for (size_t ii = n - 1; ii > 0; ii--)
    {
        size_t   jj  = (size_t) (xorshift32(rng) % (ii + 1));
        uint32_t tmp = indices[ii];
        indices[ii]  = indices[jj];
        indices[jj]  = tmp;
    }
}

/**
 * @brief Print usage information.
 */
static void print_usage(
    const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Options:\n");
    printf("  -i <name>    Input stream name (default: pts_in)\n");
    printf("  -t <name>    Ground truth stream name (default: pts_truth)\n");
    printf("  -o <path>    Output model weights file (default: model_3d.bin)\n");
    printf("  -e <epochs>  Number of training epochs (default: %d)\n", DEFAULT_EPOCHS);
    printf("  -b <size>    Batch size (default: %d)\n", DEFAULT_BATCH_SIZE);
    printf("  -r <lr>      Learning rate (default: %f)\n", DEFAULT_LR);
    printf("  -h           Show this help message\n");
}

int main(
    int    argc,
    char **argv)
{
    const char *in_name    = "pts_in";
    const char *truth_name = "pts_truth";
    const char *out_model  = "model_3d.bin";
    int         epochs     = DEFAULT_EPOCHS;
    int         batch_size = DEFAULT_BATCH_SIZE;
    float       lr         = DEFAULT_LR;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
        {
            in_name = argv[++i];
        }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
        {
            truth_name = argv[++i];
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            out_model = argv[++i];
        }
        else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc)
        {
            epochs = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
        {
            batch_size = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
        {
            lr = (float) atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    printf("=== [trtinfer_train_mlp] Neural Network C Trainer ===\n");
    printf("Connecting to input stream '%s'...\n", in_name);
    IMAGE im_in;
    memset(&im_in, 0, sizeof(IMAGE));
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &im_in) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: cannot connect to input stream '%s'\n", in_name);
        return EXIT_FAILURE;
    }

    printf("Connecting to truth stream '%s'...\n", truth_name);
    IMAGE im_truth;
    memset(&im_truth, 0, sizeof(IMAGE));
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(truth_name, &im_truth) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: cannot connect to truth stream '%s'\n", truth_name);
        ImageStreamIO_closeIm(&im_in);
        return EXIT_FAILURE;
    }

    /* Verify dimensions */
    if (im_in.md->datatype != _DATATYPE_FLOAT || im_truth.md->datatype != _DATATYPE_FLOAT)
    {
        fprintf(stderr, "Error: streams must be _DATATYPE_FLOAT\n");
        ImageStreamIO_closeIm(&im_in);
        ImageStreamIO_closeIm(&im_truth);
        return EXIT_FAILURE;
    }

    size_t nsamples = im_in.md->size[1];
    if (im_in.md->size[0] != DIM_IN || im_truth.md->size[0] != DIM_OUT ||
        im_truth.md->size[1] != nsamples)
    {
        fprintf(stderr, "Error: mismatched stream dimensions (%ux%u vs %ux%u)\n",
                im_in.md->size[0], im_in.md->size[1],
                im_truth.md->size[0], im_truth.md->size[1]);
        ImageStreamIO_closeIm(&im_in);
        ImageStreamIO_closeIm(&im_truth);
        return EXIT_FAILURE;
    }

    printf("Dataset size: %zu samples of dimension %d -> %d\n", nsamples, DIM_IN, DIM_OUT);

    const float *x_data = im_in.array.F;
    const float *y_data = im_truth.array.F;

    /* Network weights and biases */
    float w1[DIM_H1 * DIM_IN];  /* 64 x 3 */
    float b1[DIM_H1];           /* 64 */
    float w2[DIM_H2 * DIM_H1];  /* 64 x 64 */
    float b2[DIM_H2];           /* 64 */
    float w3[DIM_OUT * DIM_H2]; /* 3 x 64 */
    float b3[DIM_OUT];          /* 3 */

    /* Adam momentum buffers */
    float mw1[DIM_H1 * DIM_IN],  vw1[DIM_H1 * DIM_IN];
    float mb1[DIM_H1],           vb1[DIM_H1];
    float mw2[DIM_H2 * DIM_H1],  vw2[DIM_H2 * DIM_H1];
    float mb2[DIM_H2],           vb2[DIM_H2];
    float mw3[DIM_OUT * DIM_H2], vw3[DIM_OUT * DIM_H2];
    float mb3[DIM_OUT],          vb3[DIM_OUT];

    memset(mw1, 0, sizeof(mw1)); memset(vw1, 0, sizeof(vw1));
    memset(mb1, 0, sizeof(mb1)); memset(vb1, 0, sizeof(vb1));
    memset(mw2, 0, sizeof(mw2)); memset(vw2, 0, sizeof(vw2));
    memset(mb2, 0, sizeof(mb2)); memset(vb2, 0, sizeof(vb2));
    memset(mw3, 0, sizeof(mw3)); memset(vw3, 0, sizeof(vw3));
    memset(mb3, 0, sizeof(mb3)); memset(vb3, 0, sizeof(vb3));

    /* Initialize weights with He (Kaiming) normal */
    uint32_t rng = 123456789U;
    float std1 = sqrtf(2.0f / (float) DIM_IN);
    for (int i = 0; i < DIM_H1 * DIM_IN; i++)
    {
        w1[i] = rand_normal(&rng) * std1;
    }
    memset(b1, 0, sizeof(b1));

    float std2 = sqrtf(2.0f / (float) DIM_H1);
    for (int i = 0; i < DIM_H2 * DIM_H1; i++)
    {
        w2[i] = rand_normal(&rng) * std2;
    }
    memset(b2, 0, sizeof(b2));

    float std3 = sqrtf(2.0f / (float) DIM_H2);
    for (int i = 0; i < DIM_OUT * DIM_H2; i++)
    {
        w3[i] = rand_normal(&rng) * std3;
    }
    memset(b3, 0, sizeof(b3));

    /* Allocate batch activation and gradient buffers */
    float *act_z1 = (float *) malloc(batch_size * DIM_H1 * sizeof(float));
    float *act_a1 = (float *) malloc(batch_size * DIM_H1 * sizeof(float));
    float *act_z2 = (float *) malloc(batch_size * DIM_H2 * sizeof(float));
    float *act_a2 = (float *) malloc(batch_size * DIM_H2 * sizeof(float));
    float *act_y  = (float *) malloc(batch_size * DIM_OUT * sizeof(float));

    float *grad_w1 = (float *) malloc(DIM_H1 * DIM_IN * sizeof(float));
    float *grad_b1 = (float *) malloc(DIM_H1 * sizeof(float));
    float *grad_w2 = (float *) malloc(DIM_H2 * DIM_H1 * sizeof(float));
    float *grad_b2 = (float *) malloc(DIM_H2 * sizeof(float));
    float *grad_w3 = (float *) malloc(DIM_OUT * DIM_H2 * sizeof(float));
    float *grad_b3 = (float *) malloc(DIM_OUT * sizeof(float));

    float *delta3 = (float *) malloc(batch_size * DIM_OUT * sizeof(float));
    float *delta2 = (float *) malloc(batch_size * DIM_H2 * sizeof(float));
    float *delta1 = (float *) malloc(batch_size * DIM_H1 * sizeof(float));

    uint32_t *indices = (uint32_t *) malloc(nsamples * sizeof(uint32_t));
    for (size_t i = 0; i < nsamples; i++)
    {
        indices[i] = (uint32_t) i;
    }

    printf("Starting training: %d epochs, batch size %d, lr %f...\n", epochs, batch_size, lr);
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int timestep = 0;
    float beta1_pow = 1.0f;
    float beta2_pow = 1.0f;
    float final_rmse = 0.0f;

    for (int ep = 1; ep <= epochs; ep++)
    {
        shuffle_indices(indices, nsamples, &rng);
        size_t nbatches = nsamples / batch_size;

        for (size_t b = 0; b < nbatches; b++)
        {
            timestep++;
            beta1_pow *= BETA1;
            beta2_pow *= BETA2;

            /* 1. Forward Pass */
            for (int s = 0; s < batch_size; s++)
            {
                uint32_t idx = indices[b * batch_size + s];
                const float *px = &x_data[idx * DIM_IN];

                /* Layer 1: z1 = W1 * x + b1, a1 = ReLU(z1) */
                for (int h = 0; h < DIM_H1; h++)
                {
                    float sum = b1[h];
                    sum += w1[h * DIM_IN + 0] * px[0];
                    sum += w1[h * DIM_IN + 1] * px[1];
                    sum += w1[h * DIM_IN + 2] * px[2];
                    act_z1[s * DIM_H1 + h] = sum;
                    act_a1[s * DIM_H1 + h] = (sum > 0.0f) ? sum : 0.0f;
                }

                /* Layer 2: z2 = W2 * a1 + b2, a2 = ReLU(z2) */
                for (int h = 0; h < DIM_H2; h++)
                {
                    float sum = b2[h];
                    const float *w_row = &w2[h * DIM_H1];
                    const float *a_row = &act_a1[s * DIM_H1];
                    for (int k = 0; k < DIM_H1; k++)
                    {
                        sum += w_row[k] * a_row[k];
                    }
                    act_z2[s * DIM_H2 + h] = sum;
                    act_a2[s * DIM_H2 + h] = (sum > 0.0f) ? sum : 0.0f;
                }

                /* Layer 3: y_hat = W3 * a2 + b3 */
                for (int o = 0; o < DIM_OUT; o++)
                {
                    float sum = b3[o];
                    const float *w_row = &w3[o * DIM_H2];
                    const float *a_row = &act_a2[s * DIM_H2];
                    for (int k = 0; k < DIM_H2; k++)
                    {
                        sum += w_row[k] * a_row[k];
                    }
                    act_y[s * DIM_OUT + o] = sum;
                }

                /* Output error: delta3 = (y_hat - y) / B */
                const float *py = &y_data[idx * DIM_OUT];
                float inv_b = 1.0f / (float) batch_size;
                for (int o = 0; o < DIM_OUT; o++)
                {
                    delta3[s * DIM_OUT + o] = (act_y[s * DIM_OUT + o] - py[o]) * inv_b;
                }
            }

            /* 2. Backward Pass */
            memset(grad_w3, 0, DIM_OUT * DIM_H2 * sizeof(float));
            memset(grad_b3, 0, DIM_OUT * sizeof(float));
            for (int s = 0; s < batch_size; s++)
            {
                const float *d3 = &delta3[s * DIM_OUT];
                const float *a2 = &act_a2[s * DIM_H2];
                for (int o = 0; o < DIM_OUT; o++)
                {
                    grad_b3[o] += d3[o];
                    for (int k = 0; k < DIM_H2; k++)
                    {
                        grad_w3[o * DIM_H2 + k] += d3[o] * a2[k];
                    }
                }

                /* delta2 = (W3^T * delta3) * (z2 > 0) */
                const float *z2 = &act_z2[s * DIM_H2];
                for (int k = 0; k < DIM_H2; k++)
                {
                    float sum = 0.0f;
                    for (int o = 0; o < DIM_OUT; o++)
                    {
                        sum += w3[o * DIM_H2 + k] * d3[o];
                    }
                    delta2[s * DIM_H2 + k] = (z2[k] > 0.0f) ? sum : 0.0f;
                }
            }

            memset(grad_w2, 0, DIM_H2 * DIM_H1 * sizeof(float));
            memset(grad_b2, 0, DIM_H2 * sizeof(float));
            for (int s = 0; s < batch_size; s++)
            {
                const float *d2 = &delta2[s * DIM_H2];
                const float *a1 = &act_a1[s * DIM_H1];
                for (int h = 0; h < DIM_H2; h++)
                {
                    grad_b2[h] += d2[h];
                    for (int k = 0; k < DIM_H1; k++)
                    {
                        grad_w2[h * DIM_H1 + k] += d2[h] * a1[k];
                    }
                }

                /* delta1 = (W2^T * delta2) * (z1 > 0) */
                const float *z1 = &act_z1[s * DIM_H1];
                for (int k = 0; k < DIM_H1; k++)
                {
                    float sum = 0.0f;
                    for (int h = 0; h < DIM_H2; h++)
                    {
                        sum += w2[h * DIM_H1 + k] * d2[h];
                    }
                    delta1[s * DIM_H1 + k] = (z1[k] > 0.0f) ? sum : 0.0f;
                }
            }

            memset(grad_w1, 0, DIM_H1 * DIM_IN * sizeof(float));
            memset(grad_b1, 0, DIM_H1 * sizeof(float));
            for (int s = 0; s < batch_size; s++)
            {
                uint32_t idx = indices[b * batch_size + s];
                const float *px = &x_data[idx * DIM_IN];
                const float *d1 = &delta1[s * DIM_H1];
                for (int h = 0; h < DIM_H1; h++)
                {
                    grad_b1[h] += d1[h];
                    grad_w1[h * DIM_IN + 0] += d1[h] * px[0];
                    grad_w1[h * DIM_IN + 1] += d1[h] * px[1];
                    grad_w1[h * DIM_IN + 2] += d1[h] * px[2];
                }
            }

            /* 3. Adam Weight Updates */
            float alpha = lr * sqrtf(1.0f - beta2_pow) / (1.0f - beta1_pow);

            #define ADAM_UPDATE(w, grad, m, v, sz) \
            for (int ii = 0; ii < (sz); ii++) { \
                m[ii] = BETA1 * m[ii] + (1.0f - BETA1) * grad[ii]; \
                v[ii] = BETA2 * v[ii] + (1.0f - BETA2) * (grad[ii] * grad[ii]); \
                w[ii] -= alpha * m[ii] / (sqrtf(v[ii]) + EPSILON); \
            }

            ADAM_UPDATE(w1, grad_w1, mw1, vw1, DIM_H1 * DIM_IN);
            ADAM_UPDATE(b1, grad_b1, mb1, vb1, DIM_H1);
            ADAM_UPDATE(w2, grad_w2, mw2, vw2, DIM_H2 * DIM_H1);
            ADAM_UPDATE(b2, grad_b2, mb2, vb2, DIM_H2);
            ADAM_UPDATE(w3, grad_w3, mw3, vw3, DIM_OUT * DIM_H2);
            ADAM_UPDATE(b3, grad_b3, mb3, vb3, DIM_OUT);
            #undef ADAM_UPDATE
        }

        /* Evaluate training loss periodically */
        if (ep % 50 == 0 || ep == epochs)
        {
            double total_sq_err = 0.0;
            for (size_t i = 0; i < nsamples; i++)
            {
                const float *px = &x_data[i * DIM_IN];
                const float *py = &y_data[i * DIM_OUT];

                float h1[DIM_H1];
                for (int h = 0; h < DIM_H1; h++)
                {
                    float s = b1[h] + w1[h * DIM_IN + 0] * px[0] +
                              w1[h * DIM_IN + 1] * px[1] +
                              w1[h * DIM_IN + 2] * px[2];
                    h1[h] = (s > 0.0f) ? s : 0.0f;
                }

                float h2[DIM_H2];
                for (int h = 0; h < DIM_H2; h++)
                {
                    float s = b2[h];
                    for (int k = 0; k < DIM_H1; k++)
                    {
                        s += w2[h * DIM_H1 + k] * h1[k];
                    }
                    h2[h] = (s > 0.0f) ? s : 0.0f;
                }

                for (int o = 0; o < DIM_OUT; o++)
                {
                    float s = b3[o];
                    for (int k = 0; k < DIM_H2; k++)
                    {
                        s += w3[o * DIM_H2 + k] * h2[k];
                    }
                    float err = s - py[o];
                    total_sq_err += (double) (err * err);
                }
            }

            double mse = total_sq_err / (double) (nsamples * DIM_OUT);
            final_rmse = (float) sqrt(mse);
            printf("Epoch %3d/%d - MSE: %.6f, RMSE: %.4f\n", ep, epochs, mse, final_rmse);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double train_time = (double)(t_end.tv_sec - t_start.tv_sec) +
                        (double)(t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
    printf("Training completed in %.3f seconds (Final RMSE: %.4f)\n", train_time, final_rmse);

    /* Save model file */
    printf("Saving trained model to '%s'...\n", out_model);
    FILE *fp = fopen(out_model, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open output model file '%s' for writing\n", out_model);
    }
    else
    {
        trt_model_header_t hdr;
        hdr.magic      = TRT_MODEL_MAGIC;
        hdr.num_layers = 3;
        hdr.in_dim     = DIM_IN;
        hdr.hidden1    = DIM_H1;
        hdr.hidden2    = DIM_H2;
        hdr.out_dim    = DIM_OUT;
        hdr.final_rmse = final_rmse;
        hdr.reserved   = 0;

        fwrite(&hdr, sizeof(trt_model_header_t), 1, fp);
        fwrite(w1, sizeof(float), DIM_H1 * DIM_IN, fp);
        fwrite(b1, sizeof(float), DIM_H1, fp);
        fwrite(w2, sizeof(float), DIM_H2 * DIM_H1, fp);
        fwrite(b2, sizeof(float), DIM_H2, fp);
        fwrite(w3, sizeof(float), DIM_OUT * DIM_H2, fp);
        fwrite(b3, sizeof(float), DIM_OUT, fp);
        fclose(fp);
        printf("Model successfully written to '%s' (%zu bytes)\n",
               out_model,
               sizeof(trt_model_header_t) +
               (DIM_H1 * DIM_IN + DIM_H1 + DIM_H2 * DIM_H1 + DIM_H2 + DIM_OUT * DIM_H2 + DIM_OUT) * sizeof(float));
    }

    /* Cleanup memory and shared memory */
    free(act_z1); free(act_a1); free(act_z2); free(act_a2); free(act_y);
    free(grad_w1); free(grad_b1); free(grad_w2); free(grad_b2); free(grad_w3); free(grad_b3);
    free(delta3); free(delta2); free(delta1); free(indices);

    ImageStreamIO_closeIm(&im_in);
    ImageStreamIO_closeIm(&im_truth);

    return (final_rmse < 0.10f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
