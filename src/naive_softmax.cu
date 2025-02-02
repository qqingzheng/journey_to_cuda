#include <cuda_runtime.h>

__global__ void softmax(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;
    float max_val = input[idx];

    for (int i = 0; i < n; i++) {
        sum += exp(input[i]);
    }
    
    output[idx] = exp(input[idx]) / sum;
}

int main() {
    return 0;
}
