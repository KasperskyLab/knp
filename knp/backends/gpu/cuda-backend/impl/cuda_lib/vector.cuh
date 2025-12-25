/**
 * @file vector.cuh
 * @brief CUDA STL-like vector implemented to work on GPU.
 * @kaspersky_support Artiom N.
 * @date 06.07.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <utility>

#include <cuda_runtime.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <spdlog/spdlog.h>

#include "get_blocks_config.cuh"
#include "safe_call.cuh"
#include "cu_alloc.cuh"

#include "extraction.cuh"
#include "printf.cuh"
#include "vector_kernels.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

template <typename T, typename Allocator = CuMallocAllocator<T>>
class CUDAVector
{
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = typename allocator_type::pointer;
    using const_pointer = const typename allocator_type::pointer;
    using iterator = pointer;
    using const_iterator = const iterator;

public:
    __host__ CUDAVector(const std::initializer_list<value_type> &init_list)
        : capacity_(init_list.size()), size_(init_list.size())
    {
        if (!size_) return;
        data_ = allocator_.allocate(capacity_);
        if constexpr (std::is_trivially_copyable<value_type>::value)
        {
            call_and_check(cudaMemcpy(data_, std::data(init_list), init_list.size() * sizeof(value_type),
                                      cudaMemcpyHostToDevice));
        }
        else
        {
            for (size_t i = 0; i < init_list.size(); ++i)
            {
                gpu_insert(*(init_list.begin() + i), data_ + i); // TODO Parallelize
            }
        }
    }

    __host__ explicit CUDAVector(const std::vector<value_type> &vec) : capacity_(vec.size()), size_(vec.size())
    {
        if (!vec.size()) return;
        data_ = allocator_.allocate(capacity_);

        if constexpr (std::is_trivially_copyable<value_type>::value)
        {
            call_and_check(cudaMemcpy(data_, vec.data(), vec.size() * sizeof(value_type), cudaMemcpyHostToDevice));
        }
        else
        {
            for (size_t i = 0; i < vec.size(); ++i)
            {
                gpu_insert(vec[i], data_ + i); // TODO Parallelize
            }
        }
    }

    __host__ std::vector<value_type> to_std() const
    {
        std::vector<value_type> res;
        if (!size_) return res;
        if constexpr (std::is_trivially_copyable<value_type>::value)
        {
            res.resize(size_);
            call_and_check(cudaMemcpy(res.data(), data_, size_ * sizeof(value_type), cudaMemcpyDeviceToHost));
        }
        else
        {
            res.reserve(size_);
            for (size_t i = 0; i < size_; ++i)
            {
                res.push_back(gpu_extract(data_ + i));
            }
        }
        return res;
    }

    // TODO: Make a cheap move from gpu to cpu and back.

    __host__ __device__ explicit CUDAVector(size_type size = 0) : capacity_(size), size_(size)
    {
        if (!size_) return;
        data_ = allocator_.allocate(capacity_);
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) Allocator::construct(data_ + i);
        #else
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        construct_kernel<T, Allocator><<<num_blocks, num_threads>>>(data_, size_);
        cudaDeviceSynchronize();
        #endif
    }

    // Copy constructor from a cpu pointer-based array.
    __host__ __device__ CUDAVector(const value_type *vec, size_type size)
    {
        reserve(size);
        size_ = size;
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) allocator_.construct(data_ + i, *(vec + i));
        #else
        if constexpr (std::is_trivially_copyable_v<value_type>)
        {
            call_and_check(cudaMemcpy(data_, vec, size * sizeof(value_type), cudaMemcpyHostToDevice));
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
            {
                gpu_insert(vec[i], data_ + i); // TODO Parallelize
            }
        }
        #endif
    }

    __host__ __device__ ~CUDAVector()
    {
        if (!data_ || !capacity_) return;
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) Allocator::destroy(data_ + i);
        allocator_.deallocate(data_);
        #else
        if (size_)
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_);
            destruct_kernel<T, Allocator><<<num_blocks, num_threads>>>(data_, size_);
            cudaDeviceSynchronize();
        }
        SPDLOG_TRACE("Deallocating memory for {} elements at {}.", capacity_, reinterpret_cast<void*>(data_));
        if (capacity_) allocator_.deallocate(data_);
        #endif
    }

    // Copy constructor.
    __host__ __device__ CUDAVector(const CUDAVector& other) : capacity_(other.capacity_), size_(other.size_)
    {
        // Capacity.
        #ifdef __CUDA_ARCH__
        data_ = allocator_.allocate(capacity_);

        for (size_type i = 0; i < other.size_; ++i)
        {
            data_[i] = other.data_[i];
        }
        #else
        data_ = allocator_.allocate(capacity_);
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        SPDLOG_TRACE("Copy constructing vector of size {}", size_);
        if (num_threads > 0)
        {
            SPDLOG_TRACE("Call copy construct kernel with {} blocks and {} threads", num_blocks, num_threads);
            copy_construct_kernel<<<num_blocks, num_threads>>>(data_, size_, other.data_);
            cudaDeviceSynchronize();
        }
        #endif
    }

    // Move constructor.
    __host__ __device__ CUDAVector(CUDAVector&& other) noexcept :
        capacity_(other.capacity_), size_(other.size_), data_(other.data_)
    {
    #ifndef __CUDA_ARCH__
        SPDLOG_TRACE("Moving construct vector with data at {} and size {}",
                     reinterpret_cast<void*>(other.data_), other.size_);
    #endif
        capacity_ = other.capacity_;
        other.capacity_ = 0;

        size_ = other.size_;
        other.size_ = 0;

        data_ = other.data_;
        other.data_ = nullptr;
    #ifndef __CUDA_ARCH__
        SPDLOG_TRACE("Done moving vector");
    #endif
    }

    // Copy assignment operator.
    __host__ __device__ CUDAVector& operator=(const CUDAVector& other)
    {
        if (this == &other) return *this;
        decltype(data_) data_new = allocator_.allocate(other.size());
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < other.size_; ++i)
        {
            allocator_.construct(data_new + i);
            data_new[i] = other.data_[i];
        }
        #else
        auto [num_blocks, num_threads] = get_blocks_config(other.size());
        copy_construct_kernel<<<num_blocks, num_threads>>>(data_new, other.size(), other.data());
        #endif
        clear();
        if (capacity_) allocator_.deallocate(data_);
        capacity_ = other.size_;
        size_ = other.size_;
        data_ = data_new;
        return *this;
    }

    // Move assignment operator.
    __host__ __device__ CUDAVector& operator=(CUDAVector&& other) noexcept
    {
        if (this == &other) return *this;
        clear();
        capacity_ = other.capacity_;
        other.capacity_ = 0;
        size_ = other.size_;
        other.size_ = 0;
        allocator_.deallocate(data_);
        data_ = other.data_;
        other.data_ = nullptr;

        return *this;
    }

    __host__ CUDAVector& operator=(const std::vector<value_type> &vec)
    {
        SPDLOG_TRACE("Construct from std vector with size {}", vec.size());
        clear();
        reserve(vec.size());
        if constexpr (std::is_trivially_copyable_v<value_type>)
        {
            call_and_check(cudaMemcpy(data_, vec.data(), vec.size() * sizeof(value_type), cudaMemcpyHostToDevice));
        }
        else
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_);
            copy_construct_kernel<<<num_blocks, num_threads>>>(data_, size_, vec.data());
        }
        size_ = vec.size();
        return *this;
    }

    __host__ __device__ bool operator==(const CUDAVector &other) const
    {
        if (size_ != other.size_)
        {
        #ifndef __CUDA_ARCH__
            SPDLOG_TRACE("Operator == for cuda vector: different sizes");
        #endif
            return false;
        }
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) if (*(data_ + i) != *(other.data_ + i)) return false;
        return true;
        #else
        SPDLOG_TRACE("Equality operator for cuda vectors of size {}", size_);
        bool equal;
        bool *d_equal;
        cudaMalloc(&d_equal, sizeof(bool));
        equal_kernel<<<1, 1>>>(data_, other.data_, size_, d_equal);
        call_and_check(cudaMemcpy(&equal, d_equal, sizeof(bool), cudaMemcpyDeviceToHost));
        cudaFree(d_equal);
        SPDLOG_TRACE("Vectors equal: {}", equal);

        return equal;
        #endif
    }

    // Sets size to 0 without reallocation or changing capacity.
    __host__ __device__ void clear()
    {
        if (!size_) return;
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) allocator_.destroy(data_ + i);
        #else
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        destruct_kernel<T, Allocator><<<num_blocks, num_threads>>>(data_, size_);
        cudaDeviceSynchronize();
        #endif
        size_ = 0;
    }

    __host__ __device__ bool set(uint64_t index, const value_type &value)
    {
        if (index >= size_) return false;
        #ifdef __CUDA_ARCH__
        data_[index] = value;
        #else
        SPDLOG_TRACE("Cuda vector: setting value at index {}", index);
        gpu_insert<T>(value, data_ + index);
        #endif
        return true;
    }

    __device__ value_type& operator[](size_type index) const
    {
        return data_[index];
    }

    __host__ __device__ size_type capacity() const
    {
        return capacity_;
    }

    __host__ __device__ size_type size() const
    {
        return size_;
    }

    __host__ __device__ bool empty() const
    {
        return 0 == size_;
    }

    __host__ __device__ value_type* data()
    {
        return data_;
    }

    __host__ __device__ const value_type* data() const
    {
        return data_;
    }

    __host__ __device__ void push_back(const value_type& value)
    {
        if (size_ == capacity_) reserve((size_ + 1) * 2);
    #ifdef __CUDA_ARCH__
        Allocator::construct(data_ + size_, value);
        ++size_;
    #else
        resize(size_ + 1);
        set(size_ - 1, value);
    #endif
    }

    __device__ value_type pop_back()
    {
        if (!size_) return value_type{};
        return data_[--size_];
    }

    __host__ __device__ void reserve(size_type new_capacity)
    {
        if (new_capacity <= capacity_) return;
        #ifdef __CUDA_ARCH__
        dev_reserve(new_capacity);
        #else
        SPDLOG_TRACE("Reserving cuda vector with size {} and capacity {} for capacity {}", size_, capacity_,
                     new_capacity);
        T* new_data = allocator_.allocate(new_capacity);
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        // Inefficient, better to use move.
        if (num_threads > 0)
        {
            SPDLOG_TRACE("Running copy construct kernel with {} blocks and {} threads.", num_blocks, num_threads);
            copy_construct_kernel<<<num_blocks, num_threads>>>(new_data, size_, data_);
            cudaDeviceSynchronize();
            auto result = cudaGetLastError();
            if (result != cudaSuccess)
            {
                SPDLOG_WARN("Error: {}", cudaGetErrorString(result));
            }
            SPDLOG_TRACE("Running destruct kernel with {} blocks and {} threads", num_blocks, num_threads);
            destruct_kernel<T, Allocator><<<num_blocks, num_threads>>>(data_, size_);
            cudaDeviceSynchronize();
        }
        SPDLOG_TRACE("Data reserved, freeing old memory at {}", reinterpret_cast<const void*>(data_));
        allocator_.deallocate(data_);
        data_ = new_data;
        capacity_ = new_capacity;
        #endif
    }

    __host__ __device__ void resize(size_type new_size)
    {
        if (new_size == size_) return;
        #ifdef __CUDA_ARCH__
        dev_resize(new_size);
        #else
        if (new_size > size_)
        {
            reserve(new_size);
            auto [num_blocks, num_threads] = get_blocks_config(new_size - size_);
            construct_kernel<T, Allocator><<<num_blocks, num_threads>>>(data_ + size_, new_size - size_);
        }
        else if (new_size < size_)
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_ - new_size);
            destruct_kernel<T, Allocator><<<num_blocks, num_threads>>>(data_ + new_size, size_ - new_size);
        }
        cudaDeviceSynchronize();
        size_ = new_size;
        #endif
    }

    __host__ __device__ iterator begin() { return data_; }
    __host__ __device__ const iterator begin() const { return data_; }
    __host__ __device__ const_iterator cbegin() const { return data_; }

    __host__ __device__ iterator end() { return data_ + size_; }
    __host__ __device__ const iterator end() const { return data_ + size_; }
    __host__ __device__ const_iterator cend() const { return data_ + size_; }

    __host__ __device__ void erase(iterator begin_iter, iterator end_iter)
    {
        if (begin_iter < begin() || begin_iter >= end()) return;
        if (end_iter <= begin_iter) return;
        const size_t num_to_remove = ::cuda::std::min(end_iter, end()) - begin_iter;
        if (end_iter >= end())
        {
            resize(size_ - num_to_remove);
            return;
        }
        size_t tail_length = end() - end_iter;
        size_t num_destruct = end() - begin_iter;
#ifdef __CUDA_ARCH__
        for (size_t i = 0; i < tail_length; ++i)
        {
            *(begin_iter + i) = ::cuda::std::move(*(begin_iter + i + num_to_remove));
        }
        resize(size_ - num_destruct + tail_length);
#else
        // Allocate memory
        T* data_buf = allocator_.allocate(tail_length);
        // Move_construct tail to memory. All these kernels are implicitly synchronized.
        auto [num_blocks_mv, num_threads_mv] = get_blocks_config(tail_length);
        move_construct_kernel<T><<<num_blocks_mv, num_threads_mv>>>(data_buf, tail_length, end_iter);
        // destruct all past begin.
        auto [num_blocks_er, num_threads_er] = get_blocks_config(num_destruct);
        destruct_kernel<T, Allocator><<<num_blocks_er, num_threads_er>>>(begin_iter, num_destruct);
        // Move_construct to source vector.
        move_construct_kernel<T><<<num_blocks_mv, num_threads_mv>>>(begin_iter, tail_length, data_buf);
        resize(size_ - num_destruct + tail_length);
        // Clean up buffer
        destruct_kernel<T, Allocator><<<num_blocks_mv, num_threads_mv>>>(data_buf, tail_length);
        allocator_.deallocate(data_buf);
#endif
    }

    __host__ T copy_at(size_t index) const
    {
        if constexpr (std::is_trivially_copyable<T>::value)
        {
            T result;
            call_and_check(cudaMemcpy(&result, data_ + index, sizeof(T), cudaMemcpyDeviceToHost));
            return result;
        }
        else
            return gpu_extract(data_ + index);
    }

    /**
     * @brief Fix a vector that has been shallow-copied.
     */
    __host__ __device__ void actualize()
    {
        if (!size_)
        {
            data_ = nullptr;
            return;
        }
        T* source_data = data_;
        data_ = allocator_.allocate(capacity_);
    #ifdef __CUDA_ARCH__
        PRINTF_TRACE("Device vector actualize\n");
        for (size_t i = 0; i < size_; ++i)
            new (data_ + i) T(*(source_data + i));
    #else
        SPDLOG_TRACE("Actualizing vector of {}.", typeid(T).name());
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        copy_construct_kernel<<<num_blocks, num_threads>>>(data_, size_, source_data);
        cudaDeviceSynchronize();
        SPDLOG_TRACE("Done actualizing vector");
    #endif
    }

private:
    __device__ void dev_reserve(size_type new_capacity)
    {
        if (new_capacity <= capacity_) return;
        T* new_data = nullptr;

        // Capacity.
        new_data = allocator_.allocate(new_capacity * sizeof(T));

        for (size_type i = 0; i < size_; ++i)
        {
            new_data[i] = data_[i];
            allocator_.destroy(data_ + i);
        }
        if (data_ && capacity_)
            allocator_.deallocate(data_);

        data_ = new_data;
        capacity_ = new_capacity;
    }

    __device__ void dev_resize(size_type new_size)
    {
        if (new_size < size_)
        {
            for (size_type i = new_size; i < size_; ++i)
            {
                Allocator::destroy(data_ + i);
            }
        }
        else if (new_size > size_)
        {
            dev_reserve(new_size);
            for (size_type i = size_; i < new_size; ++i)
            {
                Allocator::construct(data_ + i);
            }
        }
        size_ = new_size;
    }

    Allocator allocator_;
    pointer data_ = nullptr;
    // Maximum elements count.
    size_type capacity_ = 0;
    // Current element.
    size_type size_ = 0;
};

} // namespace knp::backends::gpu::cuda::device_lib


namespace knp::backends::gpu::cuda
{
template<class T, class Allocator>
__host__ device_lib::CUDAVector<T, Allocator> gpu_extract<device_lib::CUDAVector<T, Allocator>>(
    const device_lib::CUDAVector<T, Allocator> *dev_vector)
{
    device_lib::CUDAVector<T, Allocator> result;

    call_and_check(cudaMemcpy(&result, dev_vector, sizeof(device_lib::CUDAVector<T, Allocator>),
                              cudaMemcpyDeviceToHost));
    result.actualize();

    return result;
}



} // namespace knp::backends::gpu::cuda
