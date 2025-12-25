/**
 * @file kernels.cu
 * @brief CUDA STL-like vector implementation to work on GPU.
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

#include "kernels.cuh"

#include <cuda_runtime.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <utility>

#include "printf.cuh"
#include "vector.cuh"
#include "safe_call.cuh"
#include "cu_alloc.cuh"
#include "../uid.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

__global__ void has_sender_kernel(UID uid, const UID *senders, size_t num_senders, int *result)
{
    uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_senders) return;
    if (senders[index] != uid) return;
    PRINTF_TRACE("Uids are equal\n");
    atomicOr(result, 1);
}

}  // namespace knp::backends::gpu::cuda::device_lib
