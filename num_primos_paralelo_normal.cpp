#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

// Função que verifica se um número é primo
bool isPrime(int num) {
   if (num <= 1)
       return false;

   for (int i = 2; i * i <= num; ++i) {  // Checa se 'num' é divisível por qualquer número até sua raiz quadrada
       if (num % i == 0)
           return false;
   }
   return true;
}

// Função que conta números primos em um intervalo dado
int countPrimesInRange(int start, int end) {
    int localCount = 0;
    // Verifica cada número no intervalo [start, end] e conta quantos são primos
    for (int i = start; i <= end; ++i) {
        if (isPrime(i))
            ++localCount;  // Incrementa o contador local se o número for primo
    }
    return localCount;
}

int main(int argc, char *argv[]) {
    // Verifica se o número correto de argumentos foi passado
    if (argc < 3) {
        std::cout << "Uso: " << argv[0] << " <Tamanho do problema> <Número de Threads>\n";
        return -1;
    }

    // Converte os argumentos para inteiros
    int interval = atoi(argv[1]);  // Intervalo de números a serem verificados
    int numThreads = atoi(argv[2]);  // Número de threads a serem utilizadas

    // Verifica se o número de threads é válido
    if (numThreads <= 0) {
        std::cerr << "O número de threads deve ser maior que zero\n";
        return -1;
    }

    // Marca o tempo de início da execução
    auto t1 = std::chrono::high_resolution_clock::now();

    // Vetores para armazenar as threads e os contadores locais de primos
    std::vector<std::thread> threads;
    std::vector<int> localCounts(numThreads, 0);  // Inicializa um contador para cada thread

    // Divide o intervalo de números de forma balanceada entre as threads
    int chunkSize = interval / numThreads;  // Tamanho de cada pedaço que cada thread vai processar
    int remaining = interval % numThreads;  // Resto para ajustar o número de threads (se não for divisível)

    int start = 0;  // Ponto de início do intervalo para a primeira thread
    
    // Criação das threads
    for (int i = 0; i < numThreads; ++i) {
        // Calcula o fim do intervalo para cada thread
        int end = start + chunkSize - 1;
        if (i < remaining) {
            end++;  // Distribui o "resto" do intervalo entre as primeiras threads
        }

        // Cria uma thread para processar o intervalo [start, end]
        threads.emplace_back([&localCounts, i, start, end]() {
            localCounts[i] = countPrimesInRange(start, end);  // Cada thread armazena seu resultado no vetor localCounts
        });

        // Atualiza o ponto de início para a próxima thread
        start = end + 1;
    }

    // Aguarda o término de todas as threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Soma os resultados locais de todas as threads para obter o número total de primos
    int totalCount = 0;
    for (int count : localCounts) {
        totalCount += count;  // Soma os resultados de cada thread
    }

    // Marca o tempo de término da execução
    auto t_end = std::chrono::high_resolution_clock::now();

    // Exibe o número total de primos encontrados e o tempo de execução
    std::cout << totalCount << " números primos entre 0 e " << interval << std::endl;
    std::cout << "Tempo de execução (seconds): " << std::chrono::duration<double>(t_end - t1).count() << std::endl;

    return 0;
}
