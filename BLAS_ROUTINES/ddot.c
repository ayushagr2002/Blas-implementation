#include <stdio.h>
#define chi(i) X[(i)*incX]
#define psi(i) Y[(i)*incY]
float cblas_ddot(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{

    float dpsum = 0.0;
#pragma omp simd
    for (int i = 0; i < N; ++i)
    {
        dpsum += X[i * incX] * Y[i * incY];
    }
    return dpsum;
}