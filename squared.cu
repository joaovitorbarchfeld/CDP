// squared.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void square(float *d_out, float *d_in, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int SIZE = 64;
    const int ARRAY_BYTES = SIZE * sizeof(float);

    // Vetor de entrada
    float h_in[SIZE], h_out[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_in[i] = float(i);
    }

    // Ponteiros de entrada e saída para a GPU
    float *d_in, *d_out;

    // Alocar memória na GPU
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    // Copiar o vetor de entrada da CPU para a GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Lançar o kernel
    int threadsPerBlock = 16;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    square<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, SIZE);

    // Copiar o resultado de volta para a CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Imprimir o resultado
    for (int i = 0; i < SIZE; i++) {
        std::cout << h_in[i] << "^2 = " << h_out[i] << std::endl;
    }

    // Liberar memória na GPU
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
