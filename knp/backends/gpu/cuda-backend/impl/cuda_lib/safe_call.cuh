/**
 * @file safe_call.cuh
 * @brief Call CUDA function and check call result.
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


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

#define call_and_check(ans) { knp::backends::gpu::cuda::device_lib::cgpu_assert((ans), __FILE__, __LINE__); }

__host__ __device__ inline void cgpu_assert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      printf("CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

}  // namespace knp::backends::gpu::cuda::device_lib
