#include <iostream>
#include <vector>
#include <chrono>
#include "concurrentQueue.h" // Inclui o ThreadPool que você forneceu

// Função que verifica se um número é primo
bool isPrime(int num) {
    if (num <= 1)
        return false;

    for (int i = 2; i * i <= num; ++i) { // Melhoria: até a raiz quadrada de num
        if (num % i == 0)
            return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " Tamanho do problema num Threads\n";
        return -1;
    }

    int interval = atoi(argv[1]);
    int numThreads = atoi(argv[2]);

    if (numThreads <= 0) {
        std::cerr << "O número de threads deve ser maior que zero\n";
        return -1;
    }

    // Inicia o pool de threads
    ThreadPool threadPool(numThreads);

    // Vetor para armazenar os futuros resultados de cada número testado
    std::vector<std::future<bool>> results;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Para cada número dentro do intervalo, enfileiramos uma tarefa para verificar se é primo
    for (int i = 0; i < interval; ++i) {
        results.emplace_back(threadPool.enqueue([](int value) {
            return isPrime(value);
        }, i));
    }

    // Contagem de primos
    int totalCount = 0;
    for (auto& future : results) {
        bool prime = future.get(); // Obtemos o resultado da verificação
        if (prime)
            ++totalCount;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    // Exibe os resultados
    std::cout << totalCount << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    return 0;
}
