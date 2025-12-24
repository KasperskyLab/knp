/**
 * @file get_blocks_config.cuh
 * @brief Get CUDA blocks count function.
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
#include <utility>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

__host__ __device__ inline ::cuda::std::pair<size_t, size_t> get_blocks_config(size_t num_total, size_t threads_per_block = 256)
{
    size_t num_threads = ::cuda::std::min(num_total, threads_per_block);
    size_t num_blocks = (num_total + threads_per_block - 1) / threads_per_block;
    return ::cuda::std::make_pair(num_blocks, num_threads);
}

} // namespace knp::backends::gpu::cuda::device_lib
