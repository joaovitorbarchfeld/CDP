#include <iostream>
#include <future>
#include <queue>
#include <thread>
#include <vector>
#include <functional>
class ThreadPool {
   public:
       ThreadPool(size_t num_threads) {
           for (size_t i = 0; i < num_threads; ++i) {
               threads_.emplace_back([this] {
                   while (true) {
                       std::function<void()> task;
                       {
                           std::unique_lock<std::mutex> lock(mutex_);
                           condition_.wait(lock, [this] {
                               return stop_ || !tasks_.empty();
                           });
                           if (stop_ && tasks_.empty()) {
                               return;
                           }
                           task = std::move(tasks_.front());
                           tasks_.pop();
                       }
                       task();
                   }
               });
           }
       }
       ~ThreadPool() {
           {
               std::unique_lock<std::mutex> lock(mutex_);
               stop_ = true;
           }
           condition_.notify_all();
           for (std::thread& thread : threads_) {
               thread.join();
           }
       }
       template<typename F, typename... Args>
       auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
           using return_type = typename std::result_of<F(Args...)>::type;
           auto task = std::make_shared<std::packaged_task<return_type()>>(
                   std::bind(std::forward<F>(f), std::forward<Args>(args)...)
           );
           std::future<return_type> result = task->get_future();
           {
               std::unique_lock<std::mutex> lock(mutex_);
               tasks_.emplace([task]() {
                   (*task)();
               });
           }
           condition_.notify_one();
           return result;
       }


   private:
       std::vector<std::thread> threads_;
       std::queue<std::function<void()>> tasks_;
       std::mutex mutex_;
       std::condition_variable condition_;
       bool stop_ = false;
};
