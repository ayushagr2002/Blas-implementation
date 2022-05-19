#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "helper.h"
#include <omp.h>
#include "stencil.h"
#define DEFAULT_VECTOR_LENGTH 2000 // million

int main(int argc, char *argv[])
{

    struct timeval calc;
    double calctime;
    int N;         
    double a = 0.8; 
    double gflops;
    float *X, *Y;
    float alpha = 2.0, beta = 3.0;

    if (argc > 1)
        N = atoi(argv[1]);
    else
        N = DEFAULT_VECTOR_LENGTH;

    int M = 500;
    int K = 100;

    FILE* fptr = fopen("result.csv", "w");
    for (int i = 3; i < 100; i += 2)
    {

        int imgrows = HDROWS;
        int imgcols = HDCOLS;
        int stencilSize = i;
        gflops = (imgrows * imgcols * 2.0f * stencilSize * stencilSize) * 1e-09;
        float *S = malloc(sizeof(float) * stencilSize * stencilSize);
        float *A = malloc(sizeof(float) * imgrows * imgcols);
        float *B = calloc(imgrows * imgcols, sizeof(float));

        RandomVector(stencilSize * stencilSize, S);
        RandomVector(imgrows * imgcols, A);
        tick(&calc);
        Stencil(A, HD, stencilSize, S, B);
        calctime = tock(&calc);
        float mem_bw = (imgrows * imgcols * 2.0f * stencilSize * stencilSize) * 4 * 1e-09 / calctime;
        printf("%d\n", i);
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
    }
}
