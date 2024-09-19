#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include "concurrentQueue.h"


// Função que verifica se um número é primo
bool isPrime(int num) {
   if (num <= 1)
       return false;


   for (int i = 2; i < num; ++i) {
       if (num % i == 0)
           return false;
   }
   return true;
}

// Função que conta números primos em um intervalo e atualiza o contador global
void countPrimesInRange(int start, int end, int& count, std::mutex& mtx) {
    int localCount = 0;
    for (int i = start; i <= end; ++i) {
        if (isPrime(i))
            ++localCount;
    }
    std::lock_guard<std::mutex> lock(mtx); // Usar um mutex para proteger o contador global
    count += localCount;
}

int main(int argc, char *argv[]) {
    // Verifica se os argumentos necessários foram fornecidos
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " Tamanho do problema num Threads\n";
        return -1;
    }

    // Converte os argumentos para inteiros
    int interval = atoi(argv[1]); // Recebe o 1º param que é o intervalo de números a serem verificados
    int numThreads = atoi(argv[2]); // Recebe o 2º param número de threads a serem usadas

    // Verifica se o número de threads é válido
    if (numThreads <= 0) {
        std::cerr << "O número de threads deve ser maior que zero\n";
        return -1;
    }

    // Marca o tempo de início
    auto t1 = std::chrono::high_resolution_clock::now();

    //Instancia um ThreadPool com o número de threads especificado no parâmetro
    ThreadPool threadPool(numThreads);
    int count = 0;
    std::mutex mtx;

    // Dividir o intervalo de números entre as threads
    int rangeSize = interval / numThreads;
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < numThreads; ++i) {
        int start = i * rangeSize;
        int end = (i == numThreads - 1) ? interval : (i + 1) * rangeSize - 1; // Se for a última thread, o intervalo vai até o final
        // Envia a tarefa para a fila. A função countPrimesInRange será executada por uma das threads do pool
        futures.push_back(threadPool.enqueue(countPrimesInRange, start, end, std::ref(count), std::ref(mtx)));
    }

    // Espera todas as threads terminarem, obtendo o resultado de cada uma delas 
    for (auto& future : futures) {
        future.get();
    }

    // Marca o tempo de fim
    auto t_end = std::chrono::high_resolution_clock::now();

    // Imprime o resultado e o tempo de execução
    std::cout << count << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl; // Imprime o tempo de execução inicial - final

    return 0;
}
