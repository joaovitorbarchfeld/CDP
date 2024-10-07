#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Função que verifica se um número é primo, executada na GPU (dispositivo)
__device__ bool isPrimeDevice(int num) {
    if (num <= 1)
        return false;

    for (int i = 2; i * i <= num; ++i) {  // Verifica se num é divisível por i, até a raiz quadrada de num
        if (num % i == 0)
            return false;
    }
    return true;
}

// Função kernel que será executada na GPU para verificar números primos em intervalos
__global__ void countPrimesSequential(int start, int end, int* primeCount) {
    // Cálculo do índice global da thread (cada thread tem um índice único dentro do grid)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcula o número total de valores a serem verificados e distribui o trabalho por thread
    int range = end - start + 1;
    int numbersPerThread = (range + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);

    // Cada thread calcula o início e fim do intervalo que deve processar
    int threadStart = start + idx * numbersPerThread;
    int threadEnd = min(threadStart + numbersPerThread - 1, end);

    int localCount = 0;
    // Cada thread verifica se os números em seu intervalo são primos
    for (int i = threadStart; i <= threadEnd; ++i) {
        if (isPrimeDevice(i)) {
            localCount++;  // Incrementa o contador local se o número for primo
        }
    }

    // Atualiza o contador global de números primos usando atomicAdd (garantindo que várias threads possam atualizar o valor de forma segura)
    atomicAdd(primeCount, localCount);
}

int main(int argc, char* argv[]) {
    // Verifica se o número correto de argumentos foi passado
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " <Tamanho do intervalo> <Threads>\n";
        return -1;
    }

    // Lê o intervalo de números a ser verificado e o número de threads por bloco
    int interval = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    int primeCountHost = 0;  // Inicializa o contador de primos no host (CPU)
    int* primeCountDevice;  // Ponteiro para o contador de primos na GPU

    // Aloca memória na GPU para o contador de primos
    cudaMalloc((void**)&primeCountDevice, sizeof(int));
    // Copia o valor inicial do contador de primos (zero) para a GPU
    cudaMemcpy(primeCountDevice, &primeCountHost, sizeof(int), cudaMemcpyHostToDevice);

    // Calcula o número de blocos no grid necessários para cobrir o intervalo
    int blocksPerGrid = (interval + threadsPerBlock - 1) / threadsPerBlock;

    // Medir o tempo de execução (início)
    auto t1 = std::chrono::high_resolution_clock::now();

    // Chama o kernel na GPU para calcular números primos em paralelo
    countPrimesSequential<<<blocksPerGrid, threadsPerBlock>>>(0, interval, primeCountDevice);

    // Sincroniza a execução da GPU (espera todas as threads terminarem)
    cudaDeviceSynchronize();

    // Medir o tempo de execução (fim)
    auto t_end = std::chrono::high_resolution_clock::now();

    // Copia o valor do contador de primos da GPU de volta para a CPU
    cudaMemcpy(&primeCountHost, primeCountDevice, sizeof(int), cudaMemcpyDeviceToHost);

    // Exibe o número total de primos encontrados e o tempo de execução
    std::cout << "Total de números primos: " << primeCountHost << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    // Libera a memória alocada na GPU
    cudaFree(primeCountDevice);

    return 0;
}
