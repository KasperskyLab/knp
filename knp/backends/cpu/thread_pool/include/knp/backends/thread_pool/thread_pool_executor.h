/**
 * @file thread_pool_executor.h
 * @brief Thread pool executor class.
 * @kaspersky_support Vartenkov A.
 * @date 08.08.2023
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
#include <utility>

#include "thread_pool_context.h"


/**
 * @brief Namespace for CPU backend executors.
 */
namespace knp::backends::cpu_executors
{
/**
 * @brief The ThreadPoolExecutor class is a definition of the interface to thread pool used for thread execution.
 * 
 * @note Use `boost::asio::post(executor, task)` to queue more tasks.\n
 * Use `join()` to wait for task execution.
 */
class ThreadPoolExecutor
{
public:
    /**
     * @brief Construct pool executor.
     * 
     * @param context thread pool context.
     * 
     * @note Lifetime of thread pool context should be at least as long as lifetime of this class object.
     */
    explicit ThreadPoolExecutor(ThreadPoolContext &context)
        : context_(context), task_count_(std::make_shared<size_t>(0))
    {
    }

    /**
     * @brief Get thread pool context.
     * 
     * @return thread pool context.
     */
    [[nodiscard]] ThreadPoolContext &context() { return context_; }

    /**
     * @brief Use functions from the `asio` library.
     * 
     * @tparam Func function type.
     * @tparam Alloc allocator type.
     * 
     * @param function function to add to task queue.
     * @param allocator allocator.
     * 
     * @note Do not call this method directly. Use `boost::asio::post` function to call the `post` method.
     */
    template <class Func, class Alloc>
    void post(Func function, const Alloc &allocator) const
    {
        auto new_task(std::allocate_shared<Task<Func>>(
            typename std::allocator_traits<Alloc>::template rebind_alloc<char>(allocator), std::move(function),
            task_count_));
        context_.post(new_task, task_count_);
    }

    /**
     * @brief Wait for all tasks to finish.
     * 
     * @note The method does not join threads.
     */
    void join() const
    {
        std::unique_lock<std::mutex> lock(context_.mutex_);
        while (*task_count_ > 0)
            if (!context_.execute_next(lock)) context_.condition_.wait(lock);
    }

    /**
     * @brief Blocking destructor that waits for all tasks to be completed.
     */
    ~ThreadPoolExecutor() { join(); }

private:
    template <class Func>
    struct Task : ThreadPoolContext::Function
    {
        explicit Task(Func func, const std::shared_ptr<size_t> &task_count) : function(std::move(func))
        {
            work_count_ = task_count;
            execute_ = [](std::shared_ptr<ThreadPoolContext::Function> &p)
            {
                Func tmp(std::move(static_cast<Task *>(p.get())->function));
                p.reset();
                tmp();
            };
        }
        Func function;
    };

    // cppcheck-suppress unusedPrivateFunction
    void on_work_started()
    {
        std::lock_guard lock(context_.mutex_);
        context_.do_work_started(task_count_);
    }

    // cppcheck-suppress unusedPrivateFunction
    void on_work_finished()
    {
        std::lock_guard lock(context_.mutex_);
        context_.do_work_finished(task_count_);  // Locks.
    }


    /**
     * @brief Empty function needed to fulfill asio::post requirements.
     */
    template <class Func, class Alloc>
    // cppcheck-suppress unusedPrivateFunction
    [[maybe_unused]] void dispatch(Func, const Alloc &) const
    {
    }

    /**
     * @brief Empty function needed to fulfill asio::post requirements.
     */
    template <class Func, class Alloc>
    // cppcheck-suppress unusedPrivateFunction
    [[maybe_unused]] void defer(Func, const Alloc &) const
    {
    }

    ThreadPoolContext &context_;
    std::shared_ptr<size_t> task_count_;
};

}  // namespace knp::backends::cpu_executors
