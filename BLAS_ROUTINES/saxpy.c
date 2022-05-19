#define chi(i) X[(i)*incX]
#define psi(i) Y[(i)*incY]

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY)
{
#pragma omp simd
    for (int i = 0; i < N; i++)
    {
        psi(i) = alpha * chi(i) + psi(i);
    }
}