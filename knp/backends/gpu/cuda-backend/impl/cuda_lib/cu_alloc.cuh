/**
 * @file cu_alloc.cuh
 * @brief CUDA allocator used cuMalloc().
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

#include <cuda_runtime.h>
#include <utility>


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

template<typename T>
struct CuMallocAllocator
{
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;

    /**
     * @brief Allocate GPU memory.
     * @param n number of objects.
     * @param hint hint (currently unused).
     * @return pointer to allocated memory.
     */
    __host__ __device__ T* allocate(size_t n, const void* hint)
    {
        T *data = nullptr;
        if (n > 0)
        {
            call_and_check(cudaMalloc(&data, n * sizeof(T)));
        }
        return data;
    }

    __host__ __device__ T* allocate(size_t n)
    {
        return allocate(n, nullptr);
    }

    template <class... Args>
    __host__ __device__ static void construct(T* p, Args&&... args)
    {
        new (p) T(std::forward<Args>(args)...);
    }

    __host__ __device__ static void destroy(T* p)
    {
        p->~T();
    }

    __host__ __device__ void deallocate(T* p, size_t n = 0)
    {
        call_and_check(cudaFree(p));
    }
};

}  // namespace knp::backends::gpu::cuda::device_lib
