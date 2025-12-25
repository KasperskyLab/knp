/**
 * @file vector_ops.cuh
 * @brief CUDA STL-like vector operators.
 * @kaspersky_support A. Vartenkov.
 * @date 08.08.2025
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

#include "vector.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

template<class T>
__host__ std::ostream &operator<<(std::ostream &stream, const CUDAVector<T> &vec)
{
    if (vec.size() == 0)
    {
        stream << "{}";
        return stream;
    }

    stream << "{";
    for (size_t i = 0; i < vec.size() - 1; ++i)
    {
        stream << vec.copy_at(i) << ", ";
    }
    stream << vec.copy_at(vec.size() - 1) << "}";
    return stream;
}

} // namespace knp::backends::gpu::cuda::device_lib
