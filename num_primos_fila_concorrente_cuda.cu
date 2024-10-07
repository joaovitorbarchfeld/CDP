#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Função que verifica se um número é primo. Esta função será executada na GPU (dispositivo)
__device__ bool isPrimeDevice(int num) {
    if (num <= 1)
        return false;

    // Verifica se 'num' é divisível por algum número até sua raiz quadrada
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0)
            return false;
    }
    return true;
}

// Kernel que será executado na GPU. Ele verifica a primalidade de números de forma concorrente
__global__ void countPrimesConcurrent(int* numbers, bool* results, int N) {
    // Calcula o índice global da thread dentro do grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Verifica se o índice está dentro dos limites e executa a verificação de primalidade
    if (idx < N) {
        results[idx] = isPrimeDevice(numbers[idx]);  // Armazena o resultado (se é primo ou não)
    }
}

int main(int argc, char* argv[]) {
    // Verifica se os argumentos necessários foram passados
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <Tamanho do intervalo> <Threads>\n";
        return -1;
    }

    // Converte os argumentos de entrada para inteiros
    int N = atoi(argv[1]);  // Tamanho do intervalo (número de números a verificar)
    int threadsPerBlock = atoi(argv[2]);  // Número de threads por bloco na GPU

    // Aloca memória na CPU para armazenar os números e os resultados
    int* host_numbers = new int[N];  // Vetor para armazenar os números
    bool* host_results = new bool[N];  // Vetor para armazenar os resultados (se é primo ou não)

    // Inicializa o vetor de números com valores de 0 a N-1
    for (int i = 0; i < N; ++i) {
        host_numbers[i] = i;
    }

    // Aloca memória na GPU para os números e os resultados
    int* dev_numbers;  // Ponteiro para armazenar os números na GPU
    bool* dev_results;  // Ponteiro para armazenar os resultados na GPU
    cudaMalloc((void**)&dev_numbers, N * sizeof(int));  // Aloca memória para os números
    cudaMalloc((void**)&dev_results, N * sizeof(bool));  // Aloca memória para os resultados

    // Copia os números da CPU para a GPU
    cudaMemcpy(dev_numbers, host_numbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define o número de blocos no grid (divide os números entre as threads disponíveis)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Medir o tempo de execução (início)
    auto t1 = std::chrono::high_resolution_clock::now();

    // Chama o kernel na GPU para processar os números (verificar quais são primos)
    countPrimesConcurrent<<<blocksPerGrid, threadsPerBlock>>>(dev_numbers, dev_results, N);

    // Sincroniza a execução da GPU (espera todas as threads terminarem)
    cudaDeviceSynchronize();

    // Medir o tempo de execução (fim)
    auto t_end = std::chrono::high_resolution_clock::now();

    // Copia os resultados da GPU de volta para a CPU
    cudaMemcpy(host_results, dev_results, N * sizeof(bool), cudaMemcpyDeviceToHost);

    // Contabiliza o número total de primos
    int primeCount = 0;
    for (int i = 0; i < N; ++i) {
        if (host_results[i]) {
            primeCount++;  // Incrementa o contador se o número for primo
        }
    }

    // Exibe o número total de primos encontrados e o tempo de execução
    std::cout << "Total de números primos: " << primeCount << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    // Libera a memória alocada na GPU
    cudaFree(dev_numbers);
    cudaFree(dev_results);

    // Libera a memória alocada na CPU
    delete[] host_numbers;
    delete[] host_results;

    return 0;
}
