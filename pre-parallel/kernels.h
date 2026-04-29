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

__global__ void backprop_delta2(float *delta3, float *W3, float *h2a, float *delta2, int H2, int CLASSES);

__global void backprop_delta1(float *delta2, float *W2, float *h1a, float *delta1, int H1, int H2);

__global__ void update_W3(float *W3, float *delta3, float *h2a, float lr, int H2, int CLASSES);

__global__ void update_b3(float *b3, float *delta3, float lr, int CLASSES);

__global__ void update_W2(float *W2, float *delta2, float *h1a, float lr, int H1, int H2);

__global__ void update_b2(float *b2, float *delta2, float lr, int H2);

__global__ void update_W1(float *W1, float *delta1, float *train_data_n, float lr, int SIZE, int H1);

__global__ void update_b1(float *b1, float *delta1, float lr, int H1);

#endif
