#include "cblas.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc)
{
    /* // DOT PRODUCT FORMULATION (i-j-p)

    if (TransA == CblasNoTrans && TransB == CblasNoTrans)
    {
        //#pragma omp simd
        #pragma omp parallel for simd
        for (int i = 0; i < M; ++i)
        {
            //#pragma omp simd
            //#pragma omp parallel for simd
            for (int j = 0; j < K; ++j)
            {
                *(C + i * ldc + j) = cblas_sdot(N, A + i * lda, 1, B + j, ldb) + beta * *(C + i * ldc + j);
            }
        }
    }
    else if (TransA == CblasNoTrans && TransB == CblasTrans)
    {
        //#pragma omp simd
        //#pragma omp parallel for simd
        for (int i = 0; i < M; ++i)
        {
            //#pragma omp simd
            //#pragma omp parallel for simd
            for (int j = 0; j < K; ++j)
            {
                *(C + i * ldc + j) = cblas_sdot(N, A + i * lda, 1, B + i * ldb, 1) + beta * *(C + i * ldc + j);
            }
        }
    }
    else if (TransA == CblasTrans && TransB == CblasNoTrans)
    {
        //#pragma omp simd
        //#pragma omp parallel for simd
        for (int i = 0; i < M; ++i)
        {
            //#pragma omp simd
            //#pragma omp parallel for simd
            for (int j = 0; j < K; ++j)
            {
                *(C + i * ldc + j) = cblas_sdot(N, A + i, lda, B + j, ldb) + beta * *(C + i * ldc + j);
            }
        }
    }
    else if (TransA == CblasTrans && TransB == CblasTrans)
    {
        //#pragma omp simd
        //#pragma omp parallel for simd
        for (int i = 0; i < M; ++i)
        {
            //#pragma omp simd
            //#pragma omp parallel for simd
            for (int j = 0; j < K; ++j)
            {
                *(C + i * ldc + j) = cblas_sdot(N, A + i, lda, B + j * ldb, 1) + beta * *(C + i * ldc + j);
            }
        }
    } */


    // USING OUTER PRODUCT


     float *temp = calloc(M * K, sizeof(float));
    float *final = calloc(M * K, sizeof(float));
   
// #pragma omp parallel shared(final) firstprivate(temp)
    {

// #pragma omp for 
        for (int p = 0; p < N; ++p)
        {
            cblas_sger(CblasRowMajor, M, K, 1, A + p, lda, B + p * ldb, 1, temp, ldc);
        }

// #pragma omp critical
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    final[i * M + j] += temp[i* K + j];
                }
            }
        }
    }

    //printf("INSIDE: %f ", final[0][0]);
#pragma omp parallel for simd
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            *(C + i * ldc + j) = alpha * final[i * K + j] + beta * *(C + i * ldc + j);
        }
    } 
    
}