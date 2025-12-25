/**
 * @file uid.cuh
 * @brief CUDA UID implementation.
 * @kaspersky_support Artiom N.
 * @date 28.03.2025
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

#include <cuda/std/array>
#include <cuda.h>
#include <knp/core/uid.h>
#include <knp/core/impexp.h>


namespace knp::backends::gpu::cuda
{
constexpr int tag_size = 16;
using UID = ::cuda::std::array<std::uint8_t, tag_size>;


/**
 * @brief Convert from knp::core::UID to cuda UID.
 * @param uid core UID.
 */
cuda::UID to_gpu_uid(const knp::core::UID &uid);


/**
 * @brief Convert from CUDA UID to knp::core::UID.
 * @param uid CUDA UID.
 */
knp::core::UID to_cpu_uid(const cuda::UID &uid);


/**
 * @brief Create new GPU UID.
 * @return new gpu UID.
 */
inline cuda::UID new_uid() { return to_gpu_uid(knp::core::UID{}); }


inline bool empty_uid(const cuda::UID& uid)
{
    for (const auto &uid_digit : uid)
    {
        if (uid_digit) return false;
    }

    return true;
}


} // namespace knp::backends::gpu::cuda
