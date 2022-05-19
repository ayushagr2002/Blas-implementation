# Blas-implementation
Implementation of BLAS (Basic Linear Algebra Subprograms) Problems of Level 1, Level 2, Level 3 in C

Basic Linear Algebra Subprograms (BLAS) is a specification that prescribes a set of low-level routines for performing common linear algebra operations such as vector addition, scalar multiplication, dot products, linear combinations, and matrix multiplication. They are the de facto standard low-level routines for linear algebra libraries; the routines have bindings for both C ("CBLAS interface") and Fortran ("BLAS interface").

Header File: https://www.netlib.org/blas/cblas.h

## Functions Implemented:
### BLAS Level 1
```
void cblas_sscal(const int N, const float alpha, float *X, const int incX);
void cblas_dscal(const int N, const double alpha, double *X, const int incX);

float  cblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);

double cblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY);
```
### BLAS Level 2
```
void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);

void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);
```
### BLAS Level 3
```
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
```
To run, follow the below steps:
- `cd BLAS_ROUTINES`
- `make`
- `./a.out <name of the routine here>`

The output shows the GFLOPS and Memory Bandwidth for different vector and Matrix Sizes.

## Stencil Computation
Applies a `k X k` stencil to an `HD` or `UHD` image.

To run, 
- `cd STENCIL`
- `make`
- `./stencil`
