/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

#ifndef KERNELS_H
#define KERNELS_H
__global__ void forward_h1(float *train_data_n, float *W1, float *b1, float *h1, float *h1a, int SIZE, int H1);


__global__ void forward_h2(float *h1a, float *W2, float *b2, float *h2, float *h2a, int H1, int H2);

__global__ void forward_out(float *h2a, float *W3, float *b3, float *out, int H2, int CLASSES);

#endif
