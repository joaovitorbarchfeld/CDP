#include <iostream>
#include <vector>
#include <thread>
// Function that checks if a number is prime
bool isPrime(int num) {
    if (num <= 1)
        return false;

    for (int i = 2; i < num; ++i) {
        if (num % i == 0)
            return false;
    }
    return true;
}
// Function to count prime numbers in a given range
int countPrimesInRange(int start, int end) {
    int count = 0;
    //std::cout << "Thread: " << std::this_thread::get_id() << " processando: " << end-start << " itens - de " << start << " até: " << end  << std::endl;
    //auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = start; i <= end; ++i) {
        if (isPrime(i))
            ++count;
    }
    //auto t_end = std::chrono::high_resolution_clock::now();
    //std::cout << "Thread: " << std::this_thread::get_id() << " finalizou em " << std::chrono::duration<double>(t_end-t1).count() << " segundos" << " -encontrou: " << count << " números primos" << std::endl;
    return count;
}

int main(int argc, char *argv[])
{
    int interval=0, threadPoolSize=0;
    if (argc<3) {
        printf("Usage: bin problemSize nthreads\n\n\n");
        return -1;
    }else{
        interval = atoi(argv[1]);
        threadPoolSize = atoi(argv[2]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const int rangeStart = 1;
    const int rangeEnd = interval;
    std::vector<std::thread> threads;
    std::vector<int> threadResults(threadPoolSize, 0);
    int chunkSize = (rangeEnd - rangeStart + 1) / threadPoolSize;
    int remaining = (rangeEnd - rangeStart + 1) % threadPoolSize;
    int start = rangeStart;
    for (int i = 0; i < threadPoolSize; ++i) {
        int end = start + chunkSize - 1;
        if (i < remaining)
            ++end;
        threads.emplace_back([start, end, i, &threadResults]() {
            threadResults[i] = countPrimesInRange(start, end);
        });
        start = end + 1;
    }
    for (auto& thread : threads) {
        thread.join();
    }
    int totalPrimes = 0;
    for (int result : threadResults) {
        totalPrimes += result;
    }
    //std::cout << "We have: " << totalPrimes << " prime numbers between 0 and " <<  interval << std::endl;
    auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Execution Time(seconds): "  <<  std::chrono::duration<double>(t_end-t1).count() << std::endl;
    return 0;
}
