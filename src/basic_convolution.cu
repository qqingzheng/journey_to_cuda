#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>

#define INPUT_CHANNEL 3
#define INPUT_HEIGHT 10240
#define INPUT_WIDTH 10240
#define KERNEL_NUMBER 16
#define KERNEL_CHANNEL 3
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define OUTPUT_HEIGHT (INPUT_HEIGHT - KERNEL_HEIGHT + 1)
#define OUTPUT_WIDTH (INPUT_WIDTH - KERNEL_WIDTH + 1)

__global__ void convolution(
    double *input,
    double *output,
    double *kernel,
    int inputHeight,
    int inputWidth,
    int inputChannel,
    int kernelNumber,
    int kernelHeight,
    int kernelWidth,
    int outputHeight,
    int outputWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Output width parallel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Output height parallel
    int z = blockIdx.z;                            // Kernel number parallel

    if (x >= outputWidth || y >= outputHeight) {
        return;
    }

    for (int c = 0; c < inputChannel; c++) {
        for (int h = 0; h < kernelHeight; h++) {
            for (int w = 0; w < kernelWidth; w++) {
                output[z * outputHeight * outputWidth + y * outputWidth + x] +=
                    input[c * inputHeight * inputWidth + (y + h) * inputWidth + (x + w)] * kernel[z * kernelHeight * kernelWidth * inputChannel + c * kernelHeight * kernelWidth + h * kernelWidth + w];
            }
        }
    }
}

int main() {
    cudaError_t cudaStatus;
    printf("Input shape: (%d, %d, %d)\n", INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH);
    printf("Kernel shape: (%d, %d, %d, %d)\n", KERNEL_NUMBER, KERNEL_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH);
    printf("Output shape: (%d, %d, %d)\n", KERNEL_NUMBER, OUTPUT_HEIGHT, OUTPUT_WIDTH);

    cudaEvent_t start_malloc, stop_malloc;
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventRecord(start_malloc);

    double *input = (double *)malloc(INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double));
    double *output = (double *)malloc(KERNEL_NUMBER * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(double));
    double *kernel = (double *)malloc(KERNEL_NUMBER * KERNEL_CHANNEL * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(double));

    omp_set_num_threads(omp_get_max_threads());
    printf("Number of threads: %d\n", omp_get_max_threads());

// Initialize input
#pragma omp parallel for simd
    for (int i = 0; i < INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH; i++) {
        input[i] = i / 100013.0;
    }

// Initialize kernel
#pragma omp parallel for simd
    for (int i = 0; i < KERNEL_NUMBER * KERNEL_CHANNEL * KERNEL_HEIGHT * KERNEL_WIDTH; i++) {
        kernel[i] = i / 100013.0;
    }

// Initialize output
#pragma omp parallel for simd
    for (int i = 0; i < KERNEL_NUMBER * OUTPUT_HEIGHT * OUTPUT_WIDTH; i++) {
        output[i] = 0;
    }

    cudaEventRecord(stop_malloc);
    cudaEventSynchronize(stop_malloc);
    float malloc_time = 0;
    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    printf("Malloc time: %.3f ms\n", malloc_time);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);

    // Load to device
    double *d_input, *d_output, *d_kernel;
    cudaStatus = cudaMalloc(&d_input, INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc input failed\n");
        return 1;
    }
    cudaStatus = cudaMalloc(&d_output, KERNEL_NUMBER * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc output failed\n");
        return 1;
    }
    cudaStatus = cudaMalloc(&d_kernel, KERNEL_NUMBER * KERNEL_CHANNEL * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc kernel failed\n");
        return 1;
    }

    cudaStatus = cudaMemcpy(d_input, input, INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy input failed\n");
        return 1;
    }
    cudaStatus = cudaMemcpy(d_kernel, kernel, KERNEL_NUMBER * KERNEL_CHANNEL * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy kernel failed\n");
        return 1;
    }

    dim3 block(16, 16, 1);                                                                                     // xyz
    dim3 grid((OUTPUT_WIDTH + block.x - 1) / block.x, (OUTPUT_HEIGHT + block.y - 1) / block.y, KERNEL_NUMBER); // grid

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution<<<grid, block>>>(d_input, d_output, d_kernel, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, KERNEL_NUMBER, KERNEL_HEIGHT, KERNEL_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %.3f ms\n", milliseconds);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Memory: Used %.2f MB, Free %.2f MB, Total %.2f MB\n",
           (total_mem - free_mem) / 1024.0 / 1024.0,
           free_mem / 1024.0 / 1024.0,
           total_mem / 1024.0 / 1024.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy back to host
    cudaMemcpy(output, d_output, KERNEL_NUMBER * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(double), cudaMemcpyDeviceToHost);

    // Output to file
    const char *filename = "output.bit";
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Can't open file %s\n", filename);
        return 1;
    }
    fwrite(output, sizeof(double), KERNEL_NUMBER * OUTPUT_HEIGHT * OUTPUT_WIDTH, fp);
    fclose(fp);

    free(input);
    free(output);
    free(kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}