#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

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

// Função que conta números primos em um intervalo e retorna o número local de primos
int countPrimesInRange(int start, int end) {
    int localCount = 0;
    for (int i = start; i <= end; ++i) {
        if (isPrime(i))
            ++localCount;
    }
    return localCount;
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

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    std::vector<int> localCounts(numThreads, 0); // Contadores locais
    int rangeSize = interval / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int start = i * rangeSize;
        int end = (i == numThreads - 1) ? interval : (i + 1) * rangeSize - 1;
        threads.emplace_back([&localCounts, i, start, end]() {
            localCounts[i] = countPrimesInRange(start, end);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Somar os contadores locais
    int totalCount = 0;
    for (int count : localCounts) {
        totalCount += count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << totalCount << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    return 0;
}
