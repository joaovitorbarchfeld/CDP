#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

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
    int interval = atoi(argv[1]); // Intervalo de números a serem verificados
    int numThreads = atoi(argv[2]); // Número de threads a serem usadas

    // Verifica se o número de threads é válido
    if (numThreads <= 0) {
        std::cerr << "O número de threads deve ser maior que zero\n";
        return -1;
    }

    // Marca o tempo de início
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    int count = 0;
    std::mutex mtx;

    // Dividir o intervalo de números entre as threads
    int rangeSize = interval / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * rangeSize;
        int end = (i == numThreads - 1) ? interval : (i + 1) * rangeSize - 1;
        // Cria e inicia uma nova thread
        threads.emplace_back(countPrimesInRange, start, end, std::ref(count), std::ref(mtx));
    }

    // Espera todas as threads terminarem
    for (auto& thread : threads) {
        thread.join();
    }

    // Marca o tempo de fim
    auto t_end = std::chrono::high_resolution_clock::now();

    // Imprime o resultado e o tempo de execução
    std::cout << count << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    return 0;
}
