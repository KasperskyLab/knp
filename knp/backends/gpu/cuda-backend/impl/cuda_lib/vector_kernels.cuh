/**
 * @file vector_kernels.cuh
 * @brief CUDA STL-like vector implemented to work on GPU.
 * @kaspersky_support A. Vartenkov.
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
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <utility>

#include "safe_call.cuh"
#include "cu_alloc.cuh"
#include "extraction.cuh"
#include "../uid.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

// TODO : Move kernels to a different .h file.
template <class T, class Allocator>
__global__ void construct_kernel(T *data, size_t num_values)
{
    if (0 == num_values) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_values) return;
    printf("Construct kernel at %p, size: %lu\n", data + i, num_values);

    Allocator::construct(data + i);
}


template <class T>
__global__ void copy_construct_kernel(T* data_to, size_t num_objects, const T* data_from)
{
    if (0 == num_objects) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Copy construct kernel: index %lu, from %p to %p\n", i, data_from + i, data_to + i);
    if (i >= num_objects) return;
    new (data_to + i) T(*(data_from + i));
}


template <class T>
__global__ void copy_kernel(T* data_to, size_t num_objects, const T* data_from)
{
    if (0 == num_objects) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;
    *(data_to + i) = *(data_from + i);
}


template <class T>
__global__ void move_kernel(T* data_to, size_t num_elements, T* data_from)
{
    if (0 == num_elements) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    *(data_to + i) = ::cuda::std::move(*(data_from + i));
}


template <class T>
__global__ void move_construct_kernel(T* data_to, size_t num_elements, T* data_from)
{
    if (0 == num_elements) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    new (data_to + i) T(std::move(*(data_from + i)));
}


template<class T, class Allocator>
__global__ void destruct_kernel(T* data, size_t num_elements)
{
    if (0 == num_elements) return;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    Allocator::destroy(data + i);
}


template<typename T>
__global__ void equal_kernel(T *data_1, const T *data_2, size_t size, bool *equal)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i) *equal = true;
    __syncthreads();
    if (i >= size) return;
    if (!(*(data_1 + i) == *(data_2 + i)))
    {
        // It's only safe if all threads writing to the same location write the same value as it's here.
        *equal = false;
    }
};

} // namespace knp::backends::gpu::cuda::device_lib
