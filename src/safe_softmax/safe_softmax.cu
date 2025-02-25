#include <cuda_runtime.h>
#include <iostream>

__global__ void safe_softmax(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x;
    if (idx >= rows) return;
    
    int start = idx * cols;
    int end = start + cols;

    // Find the max value in the row
    float max_val = input[start];
    for (int i = start + 1; i < end; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute the sum of the row
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += exp(input[i] - max_val);
    }
    
    // Compute the softmax value
    for (int i = start; i < end; i++) {
        output[i] = exp(input[i] - max_val) / sum;
    }
}

void launch_safe_softmax(int rows, int cols) {
    // Host memory
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));

    // Device memory
    float *d_input, *d_output;

    // Initialize the input
    for (int i = 0; i < rows * cols; i++) {
        input[i] = 1;
    }
    input[0] = 3;
    input[1] = 9;

    // Allocate device memory   
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy the input to the device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // <<grid, block>>
    safe_softmax<<<rows, 1>>>(d_input, d_output, rows, cols);

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the output
    for (int i = 0; i < rows * cols; i++) {
        if (i % cols == 0) printf("\n");
        printf("%f ", output[i]);
    }

    // Output to file
    // const char *filename = "output.bit";
    // FILE *fp = fopen(filename, "wb");
    // if (!fp) {
        // printf("Can't open file %s\n", filename);
        // return;
    // }
    // fwrite(output, sizeof(float), rows * cols, fp);
    // fclose(fp);

    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    launch_safe_softmax(1025, 16);
    return 0;
}
