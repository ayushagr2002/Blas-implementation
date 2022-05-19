#include <stdio.h>
#define chi(i) X[(i)*incX]
#define psi(i) Y[(i)*incY]
float cblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{

    float dpsum = 0.0;
#pragma omp simd
    for (int i = 0; i < N; ++i)
    {
        dpsum += X[i * incX] * Y[i * incY];
    }
    return dpsum;
}