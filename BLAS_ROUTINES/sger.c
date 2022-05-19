#include "cblas.h"

void cblas_sger(const enum CBLAS_ORDER Order,
                const int M, const int K, const float alpha,
                const float *X, const int incX, const float *Y,
                const int incY, float *A, const int lda)
{
#pragma omp parallel for simd
    for(int i = 0; i < M; i++)
    {
//#pragma omp parallel for simd
        for(int j = 0; j < K; j++)
        {
            A[i*lda + j] += X[i*incX] * Y[j*incY];
        }
    }
    
}