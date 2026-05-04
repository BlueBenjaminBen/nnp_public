/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

//Recreate d_relu to be called by both host and device kernel functions
__host__ __device__ static inline float d_relu(float x){
    return x > 0 ? x : 0;
}

//Recreate d_drelu to be called by both host and device kernel functions
__host__ __device__ static inline float d_drelu(float y) {
    return y > 0 ? 1 : 0;
}

//Kernel function to parallelize forwarding hidden layer 1
__global__ void forward_h1(float *train_data_n, float *W1, float *b1, float *h1, float *h1a, int input_size, int h1_size){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h1_size){
        float sum = b1[j];

        for(int i=0; i < input_size; i++){
            sum += train_data_n[i] * W1[i * h1_size + j];
        }

        h1[j] = sum;
        h1a[j] = d_relu(sum);
    }
}

//Kernel function to parallelize forwarding hidden layer 2
__global__ void forward_h2(float *h1a,float *W2, float *b2, float *h2, float *h2a, int h1_size, int h2_size){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h2_size){
        float sum = b2[j];

        for(int i=0; i < h1_size; i++){
            sum += h1a[i] * W2[i * h2_size + j];
        }

        h2[j] = sum;
        h2a[j] = d_relu(sum);
    }
}

//Kernel function to parallelize forwarding the ouput/sum
__global__ void forward_out(float*h2a, float* W3, float *b3, float *out, int h2_size, int num_classes){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k < num_classes){
        float sum = b3[k];

        for (int j = 0; j < h2_size; j++){
            sum += h2a[j] * W3[j * num_classes + k];
        }
        out[k] = sum;
    }
}

//Kernel function to parallelize computing the backpropagation of delta2
__global__ void backprop_delta2(float *delta3, float *W3, float *h2a, float *delta2, int h2_size, int num_classes){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h2_size){
        float err = 0;
        for(int k = 0; k < num_classes; k++){
            err += delta3[k] * W3[j * num_classes + k];
        }
        delta2[j] = err * d_drelu(h2a[j]); 
    }
}

//Kernel function to parallelize computing the backpropagation of delta1
__global__ void backprop_delta1(float *delta2, float *W2, float *h1a, float *delta1, int h1_size, int h2_size){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h1_size){
        float err = 0;
        for(int k = 0; k < h2_size; k++){
            err += delta2[k] * W2[j * h2_size + k];
        }
        delta1[j] = err * d_drelu(h1a[j]);
    }
}

//Kernel function to update weight3
__global__ void update_W3(float *W3, float *delta3, float *h2a, float lr, int h2_size, int num_classes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h2_size * num_classes;
    if(idx < total){
        int j = idx / num_classes;
        int k = idx % num_classes;

        W3[idx] += lr * delta3[k] * h2a[j];
    }
}

//Kernel function to update bias3
__global__ void update_b3(float *b3, float *delta3, float lr, int num_classes){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < num_classes){
        b3[j] += lr * delta3[j];
    }
}

//Kernel function to udpate weight2
__global__ void update_W2(float *W2, float *delta2, float *h1a, float lr, int h1_size, int h2_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h1_size * h2_size;
    if (idx < total){
        int j = idx / h2_size;
        int k = idx % h2_size; 
        W2[idx] += lr * delta2[k] * h1a[j];
    }
}

//Kernel function to update bias2
__global__ void update_b2(float *b2, float *delta2, float lr, int h2_size){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h2_size){
        b2[j] += lr * delta2[j];
    }
}

//Kernel function to update weight1
__global__ void update_W1(float *W1, float *delta1, float *train_data_n, float lr, int input_size, int h1_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = input_size * h1_size;
    if (idx < total){
        int j = idx / h1_size;
        int k = idx % h1_size;
        W1[idx] += lr * delta1[k] * train_data_n[j];
    }
}

//Kernel function to update bias1
__global__ void update_b1(float *b1, float *delta1, float lr, int h1_size){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h1_size){
        b1[j] += lr * delta1[j];
    }
}
