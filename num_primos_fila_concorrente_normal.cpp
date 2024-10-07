#include <iostream>
#include <vector>
#include <chrono>
#include "concurrentQueue.h" // Inclui o ThreadPool para o uso de múltiplas threads

// Função que verifica se um número é primo
bool isPrime(int num) {
    if (num <= 1)
        return false;

    for (int i = 2; i * i <= num; ++i) {  // Verifica se 'num' é divisível por algum número até sua raiz quadrada
        if (num % i == 0)
            return false;
    }
    return true;
}

// Função que conta quantos números primos existem em um intervalo
int countPrimesInRange(int start, int end) {
    int count = 0;
    for (int i = start; i <= end; ++i) {
        if (isPrime(i)) {
            ++count;  // Incrementa o contador se o número for primo
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    // Verifica se os argumentos necessários foram fornecidos
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " <Tamanho do problema> <Número de Threads>\n";
        return -1;
    }

    // Converte os argumentos para inteiros
    int interval = atoi(argv[1]);  // Tamanho do intervalo de números a ser processado
    int numThreads = atoi(argv[2]);  // Número de threads a ser usado

    // Verifica se o número de threads é válido
    if (numThreads <= 0) {
        std::cerr << "O número de threads deve ser maior que zero\n";
        return -1;
    }

    // Inicia o pool de threads com o número de threads fornecido
    ThreadPool threadPool(numThreads);

    // Vetor para armazenar os futuros resultados de cada thread
    std::vector<std::future<int>> results;

    // Calcula o tamanho do bloco de números que cada thread irá processar
    int blockSize = interval / numThreads;

    // Marca o tempo de início da execução
    auto t1 = std::chrono::high_resolution_clock::now();

    // Enfileira blocos de números para cada thread processar
    for (int i = 0; i < numThreads; ++i) {
        int start = i * blockSize;  // Ponto de início do bloco
        int end = (i == numThreads - 1) ? interval : (i + 1) * blockSize - 1;  // Ponto de fim do bloco (última thread pega o restante)
        // Adiciona a tarefa ao pool de threads
        results.emplace_back(threadPool.enqueue(countPrimesInRange, start, end));
    }

    // Contagem total de números primos
    int totalCount = 0;
    for (auto& future : results) {
        totalCount += future.get();  // Obtém o resultado de cada thread e soma ao total
    }

    // Marca o tempo de término da execução
    auto t_end = std::chrono::high_resolution_clock::now();

    // Exibe o número total de números primos encontrados e o tempo de execução
    std::cout << totalCount << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    return 0;
}
