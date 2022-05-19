#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "helper.h"
#include "cblas.h"
#include <omp.h>
#define DEFAULT_VECTOR_LENGTH 500 // million

int main(int argc, char *argv[])
{

    struct timeval calc;
    FILE* fptr = fopen("result.csv", "w");
    double calctime;
    double a = 0.8; // scalar value
    double gflops;
    float alpha = 2.0, beta = 3.0;
    char command_str[100];
    if (argc > 1)
        strcpy(command_str, argv[1]);
    else
    {
        printf("Please Enter Program\n");
        return 0;
    }

    if (strcmp(command_str, "sdot") == 0)
    {
        for (long int i = 1000000; i <= 10000000; i += 1000000)
        {
            long int n = i;
            gflops = (2 * n - 1) * 1e-09;
            float *X = malloc(sizeof(float) * n);
            float *Y = malloc(sizeof(float) * n);
            RandomVector(n, X);
            RandomVector(n, Y);
            tick(&calc);
            cblas_sdot(n, X, 1, Y, 1);
            calctime = tock(&calc);
            float mem_bw = 2 * n * 4 * 1e-09 / calctime;
            printf("Vector Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
        }
    }
    else if(strcmp(command_str, "ddot") == 0)
    {
        for (long int i = 1000000; i <= 10000000; i += 1000000)
        {
            long int n = i;
            gflops = (2 * n - 1) * 1e-09;
            double *X = malloc(sizeof(double) * n);
            double *Y = malloc(sizeof(double) * n);
            RandomVectorD(n, X);
            RandomVectorD(n, Y);
            tick(&calc);
            cblas_ddot(n, X, 1, Y, 1);
            calctime = tock(&calc);
            float mem_bw = 2 * n * 4 * 1e-09 / calctime;
            printf("Vector Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
        }
    }
    else if (strcmp(command_str, "sscal") == 0)
    {
        for (long int i = 1000000; i <= 10000000; i += 1000000)
        {
            long int n = i;
            gflops = n * 1e-09;
            float *X = malloc(sizeof(float) * n);
            RandomVector(n, X);
            tick(&calc);
            cblas_sscal(n, alpha, X, 1);
            calctime = tock(&calc);
            float mem_bw = n * 4 * 1e-09 / calctime;
            printf("Vector Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if(strcmp(command_str, "dscal") == 0)
    {   
        for (long int i = 1000000; i <= 10000000; i += 1000000)
        {
            double alpha = 2.0;
            long int n = i;
            gflops = n * 1e-09;
            double *X = malloc(sizeof(double) * n);
            RandomVectorD(n, X);
            tick(&calc);
            cblas_dscal(n, alpha, X, 1);
            calctime = tock(&calc);
            float mem_bw = n * 8 * 1e-09 / calctime;
            printf("Vector Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if (strcmp(command_str, "saxpy") == 0)
    {
        for (long int i = 1000000; i <= 10000000; i += 1000000)
        {
            long int n = i;
            gflops = 2 * n * 1e-09;
            float *X = malloc(sizeof(float) * n);
            float *Y = malloc(sizeof(float) * n);
            RandomVector(n, X);
            RandomVector(n, Y);
            tick(&calc);
            cblas_saxpy(n, alpha, X, 1, Y, 1);
            calctime = tock(&calc);
            float mem_bw = 2 * n * 4 * 1e-09 / calctime;
            printf("Vector Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if(strcmp(command_str, "daxpy") == 0)
    {
        for (long int i = 1000000; i <= 10000000; i += 1000000)
        {
            double alpha = 2.0;
            long int n = i;
            gflops = 2 * n * 1e-09;
            double *X = malloc(sizeof(double) * n);
            double *Y = malloc(sizeof(double) * n);
            RandomVectorD(n, X);
            RandomVectorD(n, Y);
            tick(&calc);
            cblas_daxpy(n, alpha, X, 1, Y, 1);
            calctime = tock(&calc);
            float mem_bw = 2 * n * 8 * 1e-09 / calctime;
            printf("Vector Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if (strcmp(command_str, "sgemv") == 0)
    {
        for (int i = 1000; i <= 10000; i += 1000)
        {
            int n = i;
            int m = i;
            gflops = (2 * m * n + 2 * m) * 1e-09;
            float *A = malloc(sizeof(float) * m * n);
            float *X = malloc(sizeof(float) * n);
            float *Y = malloc(sizeof(float) * m);
            RandomVector(m * n, A);
            RandomVector(n, X);
            RandomVector(m, Y);
            tick(&calc);
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, X, 1, beta, Y, 1);
            calctime = tock(&calc);
            float mem_bw = (m * n + n + m) * 4 * 1e-09 / calctime;
            printf("Matrix Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if(strcmp(command_str, "dgemv") == 0)
    {
        for (int i = 1000; i <= 10000; i += 1000)
        {
            double alpha = 2.0, beta = 3.0;
            int n = i;
            int m = i;
            gflops = (2 * m * n + 2 * m) * 1e-09;
            double *A = malloc(sizeof(double) * m * n);
            double *X = malloc(sizeof(double) * n);
            double *Y = malloc(sizeof(double) * m);
            RandomVectorD(m * n, A);
            RandomVectorD(n, X);
            RandomVectorD(m, Y);
            tick(&calc);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, X, 1, beta, Y, 1);
            calctime = tock(&calc);
            float mem_bw = (m * n + n + m) * 8 * 1e-09 / calctime;
            printf("Matrix Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if (strcmp(command_str, "sgemm") == 0)
    {
        for (int i = 100; i <= 1000; i += 100)
        {
            int m = i;
            int n = i;
            int k = i;
            gflops = (2 * m * k * n + 2 * m * k) * 1e-09;
            float *A = malloc(sizeof(float) * m * n);
            float *B = malloc(sizeof(float) * n * k);
            float *C = malloc(sizeof(float) * m * k);
            RandomVector(m * n, A);
            RandomVector(n * k, B);
            RandomVector(m * k, C);
            tick(&calc);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, n, B, k, beta, C, k);
            calctime = tock(&calc);
            float mem_bw = ((m * n) + (n * k) + (k * n)) * 4 * 1e-09 / calctime;
            printf("Matrix Size: %7d  Throughput: %7lf GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else if(strcmp(command_str, "dgemm") == 0)
    {
        for (int i = 100; i <= 1000; i += 100)
        {
            double alpha = 2.0, beta = 3.0;
            int m = i;
            int n = i;
            int k = i;
            gflops = (2 * m * k * n + 2 * m * k) * 1e-09;
            double *A = malloc(sizeof(double) * m * n);
            double *B = malloc(sizeof(double) * n * k);
            double *C = malloc(sizeof(double) * m * k);
            RandomVectorD(m * n, A);
            RandomVectorD(n * k, B);
            RandomVectorD(m * k, C);
            tick(&calc);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, n, B, k, beta, C, k);
            calctime = tock(&calc);
            float mem_bw = ((m * n) + (n * k) + (k * n)) * 8 * 1e-09 / calctime;
            printf("Matrix Size: %7d  Throughput: %7f GFLOPS/s Time:%7f ms Memory Bandwidth:%7f GB/s\n\n", i, gflops / calctime, calctime * 1000, mem_bw);
            fprintf(fptr, "%f, %f, %f, %d\n", gflops /calctime, calctime * 1000, mem_bw, i);
        }
    }
    else
    {
        printf("You have Entered Invalid Command\n");
    }
}
