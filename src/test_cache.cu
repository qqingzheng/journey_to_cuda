#include <cuda_runtime.h>
#include <iostream>

#define N (4294967296 / 4)

__global__ void cache_test(int *data, int stride)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int accessIdx = ((long long)idx * stride) % N;
    if (accessIdx < N)
    {
        data[accessIdx] += 1;
    }
}

int main()
{
    cudaError_t cudaStatus;
    int *d_data;
    cudaStatus = cudaMalloc(&d_data, N * sizeof(int));
    printf("Memory allocated: %f MB\n", (float)N * sizeof(int) / 1024 / 1024);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMalloc failed, %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaMemset(d_data, 0, N * sizeof(int));
    for (int stride = 1; stride <= 1024 * 1024 / sizeof(int); stride *= 10)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        int blockSize = 32;
        int gridSize = (N + blockSize - 1) / blockSize;
        printf("stride: %d, gridSize: %d, blockSize: %d\n", stride, gridSize, blockSize);
        cache_test<<<gridSize, blockSize>>>(d_data, stride);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time: %.3f ms\n", milliseconds);
    }
    cudaFree(d_data);
    return 0;
}