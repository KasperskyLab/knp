/**
 * @file vector_view.cuh
 * @brief Plain old data representation for an std-like vector.
 * @kaspersky_support A. Vartenkov
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

namespace knp::backends::gpu::cuda::device_lib
{
using LongIndex = unsigned long long;

/**
 * @brief Plain Old Data structure for CUDAVector, used to pass into kernels.
 * @tparam T Value type for CUDAVector.
 */
template<class T>
struct CUDAVectorView
{
    const T *const data_;
    const LongIndex size_;
};


/**
 * @brief Plain Old Data structure for CUDAVector representation, use to pass into kernels that change the contents.
 * @tparam T value type for CUDAVector.
 */
template<class T>
struct CUDAVectorMutableView
{
    T *const data_;
    const LongIndex size_;
};
}
