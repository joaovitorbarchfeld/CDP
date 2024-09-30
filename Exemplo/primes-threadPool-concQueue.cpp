#include <iostream>
#include <vector>
#include <chrono>
#include "concurrentQueue.h"
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
    ThreadPool pool(threadPoolSize);
    std::vector<std::future<bool>> results;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i <  interval; ++i) {
        results.emplace_back(pool.enqueue([](int value) {     
            if (value <= 1)
            return false;

            // Check from 2 to n-1
            for (int i = 2; i < value; i++){
            if (value % i == 0)
                return false;
            }
            return true; 
        }, i));
    }
    int primerCount = 0;
    for (auto& result : results) {
        bool isPrime =  result.get();
        if (isPrime)
        {
           primerCount++;
        }      
    }
    //std::cout << "We have: " << primerCount << " prime numbers between 0 and " <<  interval << std::endl;
    auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Execution Time(seconds): "  <<  std::chrono::duration<double>(t_end-t_start).count() << std::endl;
    return 0;
}
