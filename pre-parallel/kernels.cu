/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

__global__ void foward_h1(float *train_data_n, float *W1, float *b1, float *h1, float *h1a, int SIZE, int H1){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < H1){
        float sum = b1[j];

        for(int i=0; i < SIZE; i++){
            sum += train_data_n[i] * W1[i * H1 + j];
        }

        h1[j] = sum;
        h1a[j] = relu(sum);
    }
}

__global__ void forward_h2(float *h1a,float *W2, float *b2, float *h2, float *h2a, int H1 int H2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < H2){
        float sum = b2[j];

        for(int i=0; i < H1; i++){
            sum += h1a[i] * W2[i * H2 + j];
        }

        h2[j] = sum;
        h2a[j] = relu(sum);
    }
}

__global__ void forward_out(float*h2a, float* W3, *b3, float *out, int H2, int CLASSES){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k < CLASSES){
        float sum = b3[k];

        for (int j = 0; j < H2; j++){
            sum += h2a[j] * W3[j * CLASSES + k];
        }
        out[k] = sum;
    }
}
