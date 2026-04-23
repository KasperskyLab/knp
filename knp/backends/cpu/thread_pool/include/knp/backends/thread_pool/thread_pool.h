/**
 * @file thread_pool.h
 * @brief Reusable thread pool class, a modified example from asio documentation that aggregates context and executor.
 * @kaspersky_support Vartenkov A.
 * @date 27.07.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <memory>

#include "thread_pool_context.h"
#include "thread_pool_executor.h"


/**
 * @brief Namespace for CPU backend executors.
 */
namespace knp::backends::cpu_executors
{
/**
 * @brief The ThreadPool class is a definition of thread pool.
 */
class ThreadPool
{
public:
    /**
     * @brief Create thread pool.
     * 
     * @param num_threads number of worker threads in the pool.
     */
    explicit ThreadPool(size_t num_threads)
        : context_(std::make_unique<ThreadPoolContext>(num_threads)), executor_(*context_)
    {
    }

    /**
     * @brief Add task to pool.
     * 
     * @tparam Func function type.
     * @tparam Args function arguments.
     * 
     * @param func task to run in the pool.
     * @param args function arguments (if required, use `std::ref`).
     * 
     * @note Non-blocking method.
     */
    template <class Func, typename... Args>
    void post(Func func, Args... args)
    {
        boost::asio::post(executor_, std::bind(func, args...));
    }

    /**
     * @brief Wait until all threads stop processing.
     * 
     * @note Blocking method that waits indefinitely if at least one task never stops.
     */
    void join() { executor_.join(); }

private:
    // Do not change the order of declarations.
    std::unique_ptr<ThreadPoolContext> context_;
    ThreadPoolExecutor executor_;
};

}  // namespace knp::backends::cpu_executors
