/**
 * @file register_all.cuh
 * @brief Macro for all types vector registration.
 * @kaspersky_support A. Vartenkov.
 * @date 06.04.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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
#include "register_type.cuh"

// TODO: make a recursive macro put of it.
#define REGISTER_ALL_TYPES \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::PopulationVariants); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::ProjectionVariants); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse); \
REGISTER_CUDA_VECTOR_TYPE(unsigned long long); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID)
