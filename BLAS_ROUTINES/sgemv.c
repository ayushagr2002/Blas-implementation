#include "cblas.h"
#include <stdio.h>
#include <malloc.h>
void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY)
{
    if (TransA == CblasNoTrans)
    {
        if (order == CblasRowMajor)
        {
#pragma omp parallel for simd
            for (int i = 0; i < M; ++i)
            {
                Y[i] = alpha * cblas_sdot(N, A + i * lda, 1, X, incX) + beta * Y[i];
            }
        }
        else
        {
            float *temp = calloc(M, sizeof(float));
            float *final = calloc(M, sizeof(float));

#pragma omp parallel shared(final) firstprivate(temp)
            {
#pragma omp for
                for (int i = 0; i < N; ++i)
                {
                    cblas_saxpy(M, X[i], A + i * M, 1, temp, 1);
                }
#pragma omp critical
                for (int i = 0; i < M; ++i)
                {
                    final[i] += temp[i];
                }
            }

#pragma omp parallel for simd
            for (int i = 0; i < M; i++)
            {
                Y[i] = alpha * final[i] + beta * Y[i];
            }

        }
    }
    else
    {
        if (order == CblasRowMajor)
        {
            float *temp = calloc(N, sizeof(float));
            float *final = calloc(N, sizeof(float));
            
#pragma omp parallel shared(final) firstprivate(temp)
            {
#pragma omp for
                for (int i = 0; i < M; ++i)
                {
                    cblas_saxpy(N, X[i], A + i * N, 1, temp, 1);
                }
#pragma omp critical
                for (int i = 0; i < N; ++i)
                {
                    final[i] += temp[i];
                }
            }

#pragma omp parallel for simd
            for (int i = 0; i < N; i++)
            {
                Y[i] = alpha * final[i] + beta * Y[i];
            }
            /* #pragma omp parallel for simd
                        for (int i = 0 ; i < N; ++i)
                        {
                            Y[i] = alpha * cblas_sdot(M, A + i, lda, X, incX) + beta * Y[i];
                        } */
        }
        else
        {
#pragma omp parallel for simd
            for (int i = 0; i < N; ++i)
            {
                Y[i] = alpha * cblas_sdot(M, A + i * lda, 1, X, incX) + beta * Y[i];
            }
        }
    }
}