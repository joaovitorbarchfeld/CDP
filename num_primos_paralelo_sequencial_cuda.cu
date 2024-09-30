#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__device__ bool isPrimeDevice(int num) {
    if (num <= 1)
        return false;

    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0)
            return false;
    }
    return true;
}

// Kernel function para verificar números primos em blocos de trabalho
__global__ void countPrimesSequential(int start, int end, int* primeCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Índice global da thread

    int range = end - start + 1;
    int numbersPerThread = (range + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);

    int threadStart = start + idx * numbersPerThread;
    int threadEnd = min(threadStart + numbersPerThread - 1, end);

    int localCount = 0;
    for (int i = threadStart; i <= threadEnd; ++i) {
        if (isPrimeDevice(i)) {
            localCount++;
        }
    }

    atomicAdd(primeCount, localCount);  // Atualização concorrente do contador de primos
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " <Tamanho do intervalo> <Threads>\n";
        return -1;
    }

    int interval = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    int primeCountHost = 0;
    int* primeCountDevice;

    cudaMalloc((void**)&primeCountDevice, sizeof(int));
    cudaMemcpy(primeCountDevice, &primeCountHost, sizeof(int), cudaMemcpyHostToDevice);

    // Definir dimensões do grid e blocos
    int blocksPerGrid = (interval + threadsPerBlock - 1) / threadsPerBlock;

    // Medir o tempo de execução
    auto t1 = std::chrono::high_resolution_clock::now();

    // Chamar o kernel na GPU para calcular primos em paralelo
    countPrimesSequential<<<blocksPerGrid, threadsPerBlock>>>(0, interval, primeCountDevice);

    // Sincronizar GPU
    cudaDeviceSynchronize();

    // Medir o tempo de execução após a execução do kernel
    auto t_end = std::chrono::high_resolution_clock::now();

    // Copiar o resultado de volta para a CPU
    cudaMemcpy(&primeCountHost, primeCountDevice, sizeof(int), cudaMemcpyDeviceToHost);

    // Exibir o número de primos e o tempo de execução
    std::cout << "Total de números primos: " << primeCountHost << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    // Liberar a memória na GPU
    cudaFree(primeCountDevice);

    return 0;
}
