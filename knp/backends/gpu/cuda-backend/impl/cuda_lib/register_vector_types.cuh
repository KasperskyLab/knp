/**
 * @file register_vector_types.cuh
 * @brief Macros for CUDA vector type registration.
 * @kaspersky_support A. Vartenkov.
 * @date 17.12.2025
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

#define REGISTER_CUDA_VECTOR_NO_EXTRACT(data_type) \
    template __global__ void knp::backends::gpu::cuda::device_lib::construct_kernel<data_type, \
    knp::backends::gpu::cuda::device_lib::CuMallocAllocator<data_type>>(data_type *, size_t); \
    template __global__ void knp::backends::gpu::cuda::device_lib::copy_construct_kernel<data_type>( \
    data_type *, size_t, const data_type *); \
    template __global__ void knp::backends::gpu::cuda::device_lib::copy_kernel<data_type>(     \
    data_type *, size_t, const data_type *);                                         \
    template __global__ void knp::backends::gpu::cuda::device_lib::move_kernel<data_type>(       \
    data_type *, size_t, data_type*);        \
    template __global__ void knp::backends::gpu::cuda::device_lib::move_construct_kernel<data_type>( \
    data_type *, size_t, data_type *);                                          \
    template __global__ void knp::backends::gpu::cuda::device_lib::destruct_kernel<data_type,    \
    knp::backends::gpu::cuda::device_lib::CuMallocAllocator<data_type>>(data_type*, size_t)


#define REGISTER_CUDA_VECTOR_TYPE(data_type) \
    REGISTER_CUDA_VECTOR_NO_EXTRACT(data_type); \
    template __host__ data_type knp::backends::gpu::cuda::gpu_extract<data_type>(const data_type* );       \
    template __host__ void knp::backends::gpu::cuda::gpu_insert<data_type>(const data_type &, data_type *)

