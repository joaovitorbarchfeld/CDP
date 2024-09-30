#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__device__ bool isPrimeDevice(int num) {
    if (num <= 1)
        return false;
    
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0)
            return false;
    }
    return true;
}

// Kernel function para calcular números primos de forma concorrente
__global__ void countPrimesConcurrent(int* numbers, bool* results, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Índice global da thread

    if (idx < N) {
        results[idx] = isPrimeDevice(numbers[idx]);  // Verificar primalidade
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <Tamanho do intervalo> <Threads>\n";
        return -1;
    }

    int N = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    // Alocar memória na CPU
    int* host_numbers = new int[N];
    bool* host_results = new bool[N];

    // Inicializar os números no intervalo
    for (int i = 0; i < N; ++i) {
        host_numbers[i] = i;
    }

    // Alocar memória na GPU
    int* dev_numbers;
    bool* dev_results;
    cudaMalloc((void**)&dev_numbers, N * sizeof(int));
    cudaMalloc((void**)&dev_results, N * sizeof(bool));

    // Copiar dados da CPU para a GPU
    cudaMemcpy(dev_numbers, host_numbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Definir dimensões do grid e blocos
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Medir o tempo de execução
    auto t1 = std::chrono::high_resolution_clock::now();

    // Chamar o kernel na GPU para processar números individualmente
    countPrimesConcurrent<<<blocksPerGrid, threadsPerBlock>>>(dev_numbers, dev_results, N);

    // Sincronizar GPU
    cudaDeviceSynchronize();

    // Medir o tempo de execução após a execução do kernel
    auto t_end = std::chrono::high_resolution_clock::now();

    // Copiar os resultados de volta para a CPU
    cudaMemcpy(host_results, dev_results, N * sizeof(bool), cudaMemcpyDeviceToHost);

    // Contar o número total de primos
    int primeCount = 0;
    for (int i = 0; i < N; ++i) {
        if (host_results[i]) {
            primeCount++;
        }
    }

    // Exibir o número de primos e o tempo de execução
    std::cout << "Total de números primos: " << primeCount << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    // Liberar a memória na GPU e CPU
    cudaFree(dev_numbers);
    cudaFree(dev_results);
    delete[] host_numbers;
    delete[] host_results;

    return 0;
}
