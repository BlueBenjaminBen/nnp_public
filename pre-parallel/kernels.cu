/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */


__host__ __device__ float relu(float x){
    return x > 0 ? x : 0;
}

__host__ __device__ float drelu(float y) {
    return y > 0 ? 1 : 0;
}

__global__ void forward_h1(float *train_data_n, float *W1, float *b1, float *h1, float *h1a, int SIZE, int H1){
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

__global__ void forward_h2(float *h1a,float *W2, float *b2, float *h2, float *h2a, int H1, int H2){
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

__global__ void forward_out(float*h2a, float* W3, float *b3, float *out, int H2, int CLASSES){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k < CLASSES){
        float sum = b3[k];

        for (int j = 0; j < H2; j++){
            sum += h2a[j] * W3[j * CLASSES + k];
        }
        out[k] = sum;
    }
}


__global__ void backprop_delta2(float *delta3, float *W3, float *h2a, float *delta2, int H2, int CLASSES){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < H2){
        float err = 0;
        for(int k = 0; k < CLASSES; k++){
            err += delta3 * W3[j * CLASSES + k];
        }
        delta2[j] = err * drelu(h2a[j]); 
    }
}

__global__ void backprop_delta1(float *delta2, float *W2, float *h1a, float *delta1, int H1, int H2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < H1){
        float err = 0;
        for(int k = 0; k < H2; k++){
            err += delta2 * W2[j * H2 + k];
        }
        delta1[j] == err * drelu(h1a[j]);
    }
}

__global__ void update_W3(float *W3, float *delta3, float *h2a, float lr, int H2, int CLASSES){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H2 * CLASSES;
    if(idx < total){
        int j = idx / CLASSES;
        int k = idx % CLASSES;

        W3[idx] += lr * delta3[k] * h2a[j];
    }
}

__global__ void update_b3(float *b3, float *delta3, float lr, int CLASSES){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < CLASSES){
        b3[j] += lr * delta3[j];
    }
}

__global__ void update_W2(float *W2, float *delta2, float *h1a, float lr, int H1, int H2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H1 * H2;
    if (idx < total){
        int j = idx / H2;
        int k = idx % H2; 
        W2[idx] += lr * delta2[k] * h1a[j];
    }
}

__global__ void update_b2(float *b2, float *delta2, float lr, int H2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < H2){
        b2[j] += lr * delta2[j];
    }
}

__global__ void update_W1(float *W1, float *delta1, float *train_data_n, float lr, int SIZE, int H1){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SIZE * H1;
    if (idx < total){
        int j = idx / H1;
        int k = idx % H1;
        W1[idx] += lr * delta1[j] * train_data_n[i];
    }
}

__global__ void update_b1(float *b1, float *delta1, float lr, int H1){
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int (j < H1){
            b1[j] += lr * delta1[j];
        }
}
