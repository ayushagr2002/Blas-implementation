#define chi(i) X[(i)*incX]
#define psi(i) Y[(i)*incY]

void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY)
{
#pragma omp simd
    for (int i = 0; i < N; i++)
    {
        psi(i) = alpha * chi(i) + psi(i);
    }
}