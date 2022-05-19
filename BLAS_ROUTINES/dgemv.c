#include "cblas.h"
#include <stdio.h>
#include <malloc.h>
void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
{
    if (TransA == CblasNoTrans)
    {
        if (order == CblasRowMajor)
        {
#pragma omp parallel for simd
            for (int i = 0; i < M; ++i)
            {
                Y[i] = alpha * cblas_ddot(N, A + i * lda, 1, X, incX) + beta * Y[i];
            }
        }
        else
        {
            double *temp = calloc(M, sizeof(double));
            double *final = calloc(M, sizeof(double));

#pragma omp parallel shared(final) firstprivate(temp)
            {
#pragma omp for
                for (int i = 0; i < N; ++i)
                {
                    cblas_daxpy(M, X[i], A + i * M, 1, temp, 1);
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
            double *temp = calloc(N, sizeof(double));
            double *final = calloc(N, sizeof(double));

#pragma omp parallel shared(final) firstprivate(temp)
            {
#pragma omp for
                for (int i = 0; i < M; ++i)
                {
                    cblas_daxpy(N, X[i], A + i * N, 1, temp, 1);
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
                Y[i] = alpha * cblas_ddot(M, A + i * lda, 1, X, incX) + beta * Y[i];
            }
        }
    }
}