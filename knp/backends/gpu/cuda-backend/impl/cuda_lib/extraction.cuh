/**
 * @file extraction.cuh
 * @brief Functions for GPU-host exchange of nontrivial types.
 * @kaspersky_support A. Vartenkov.
 * @date 23.09.2025
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
#include "safe_call.cuh"
#include "cu_alloc.cuh"

/**
 * @brief Namespace for CUDA implementations.
 */
namespace knp::backends::gpu::cuda
{
// TODO Add move extraction and insertion
template<class T>
__host__ T gpu_extract(const T* gpu_val)
{
    T result;
    call_and_check(cudaMemcpy(&result, gpu_val, sizeof(T), cudaMemcpyDeviceToHost));
    if constexpr (!std::is_trivially_copyable<T>::value)
        result.actualize();
    cudaDeviceSynchronize();
    return result;
}


namespace detail
{
template<class T>
__global__ void actualize_kernel(T *val)
{
    val->actualize();
}
} // namespace detail


template<class T>
__host__ void gpu_insert(const T &cpu_val, T *gpu_target)
{
    call_and_check(cudaMemcpy(gpu_target, &cpu_val, sizeof(T), cudaMemcpyHostToDevice));
    if constexpr (!std::is_trivially_copyable<T>::value)
        detail::actualize_kernel<<<1, 1>>>(gpu_target);
    cudaDeviceSynchronize();
}

} // namespace knp::backends::gpu::cuda
