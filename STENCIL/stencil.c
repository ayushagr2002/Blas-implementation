#include "stencil.h"
#include <malloc.h>
void Stencil(float *X, enum ImageType typ, int k, float *S, float *Y)
{

    int imgrows, imgcols;
    if(typ == UHD)
    {
        imgrows = UHDROWS;
        imgcols = UHDCOLS;
    }
    else
    {
        imgrows = HDROWS;
        imgcols = HDCOLS;
    }

    int padding = 2 * (k/2);
    
    float *padded_X = malloc(sizeof(float) * (imgrows + padding) * (imgcols + padding));

    for(int i = 0; i < imgrows + padding; ++i)
    {
        for(int j = 0; j < imgcols + padding; ++j)
        {
            padded_X[i * (imgcols + padding) + j] = 0;
        }
    }


    for(int i = padding/2; i < imgrows + padding/2; i++)
    {
        for(int j = padding/2; j < imgcols + padding/2; j++)
        {
            padded_X[i * (imgcols+padding) + j] = X[(i-padding/2) * imgcols + (j-padding/2)];
        }
    }

    /* for(int i = 0; i < imgrows + padding; ++i)
    {
        for(int j = 0; j < imgcols + padding; ++j)
        {
            printf("%f ", padded_X[i * (imgcols + padding) + j]);
        }
        printf("\n");
    } */
    // assuming k is odd

//#pragma omp parallel for simd
    for(int i = padding/2; i < imgrows + padding/2; i++)
    {
#pragma omp parallel for simd
        for(int j = padding/2; j < imgcols + padding/2; j++)
        {
            //apply stencil
            //#pragma omp parallel for simd
            for(int ii = 0; ii < k; ii++)
            {
                //#pragma omp simd
                //#pragma omp parallel for simd
                for(int jj = 0; jj < k; jj++)
                {
                    Y[(i - padding/2) * imgcols + (j - padding/2)] += S[ii*k + jj] * padded_X[(i - padding/2 + ii) * (imgcols + padding) + (j - padding/2 + jj)];
                }
            }
        }
    }
}