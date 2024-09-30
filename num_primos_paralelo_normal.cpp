#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

// Função que verifica se um número é primo
bool isPrime(int num) {
   if (num <= 1)
       return false;

   for (int i = 2; i * i <= num; ++i) {  // Checa até a raiz quadrada de 'num'
       if (num % i == 0)
           return false;
   }
   return true;
}

// Função que conta números primos em um intervalo
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
        std::cout << "Uso: " << argv[0] << " <Tamanho do problema> <Número de Threads>\n";
        return -1;
    }

    int interval = atoi(argv[1]);
    int numThreads = atoi(argv[2]);

    if (numThreads <= 0) {
        std::cerr << "O número de threads deve ser maior que zero\n";
        return -1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Vetores para armazenar threads e os resultados locais de cada thread
    std::vector<std::thread> threads;
    std::vector<int> localCounts(numThreads, 0);

    // Divisão do intervalo de forma balanceada entre as threads
    int chunkSize = interval / numThreads;
    int remaining = interval % numThreads;  // Para distribuir o restante

    int start = 0;
    
    for (int i = 0; i < numThreads; ++i) {
        // Ajustar o tamanho da última thread para cobrir todo o intervalo
        int end = start + chunkSize - 1;
        if (i < remaining) {
            end++;  // Distribuir sobras entre as primeiras threads
        }

        // Criar uma thread para processar o intervalo [start, end]
        threads.emplace_back([&localCounts, i, start, end]() {
            localCounts[i] = countPrimesInRange(start, end);
        });

        start = end + 1;  // Atualizar o ponto de início para a próxima thread
    }

    // Aguardar o término de todas as threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Somar os resultados de todas as threads
    int totalCount = 0;
    for (int count : localCounts) {
        totalCount += count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    // Exibir o número total de primos encontrados e o tempo de execução
    std::cout << totalCount << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    return 0;
}
