/**
 * @file projections_impl.cu
 * @brief Contains functions for projection calculation.
 * @kaspersky_support A. Vartenkov
 * @date 14.08.2025
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

#include "cuda_bus/message_bus.cuh"
#include "projection.cuh"
#include <boost/mp11.hpp>
#include <cuda_runtime.h>

/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
using StepIndex = unsigned long long;

/**
 * @brief List of synapse types supported by the CUDA backend.
 */
using SupportedSynapses = boost::mp11::mp_list<knp::synapse_traits::DeltaSynapse>;

/**
 * @brief List of supported projection types based on synapse types specified in `SupportedSynapses`.
 */
using SupportedProjections = boost::mp11::mp_transform<CUDAProjection, SupportedSynapses>;


template <class Projection>
__host__ __device__ inline bool is_forcing()
{
    // TODO: static_assert(false)
    return false;
}


template <>
__host__ __device__ inline bool is_forcing<CUDAProjection<knp::synapse_traits::DeltaSynapse>>() { return true; }


/**
 * @brief Projection variant that contains any projection type specified in `SupportedProjections`.
 * @details `ProjectionVariants` takes the value of `std::variant<ProjectionType_1,..., ProjectionType_n>`, where
 * `ProjectionType_[1..n]` is the projection type specified in `SupportedProjections`. \n
 * For example, if `SupportedProjections` contains DeltaSynapse and AdditiveSTDPSynapse types,
 * then `ProjectionVariants = std::variant<DeltaSynapse, AdditiveSTDPSynapse>`. \n
 * `ProjectionVariants` retains the same order of message types as defined in `SupportedProjections`.
 * @see ALL_SYNAPSES.
 */
using ProjectionVariants = boost::mp11::mp_rename<SupportedProjections, ::cuda::std::variant>;


/**
 * @brief Calculate projection of delta synapses.
 * @note Projection will be changed during calculation.
 * @param projection projection to calculate.
 * @param message_queue message queue to send to projection for calculation.
 */
__host__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection,
        const CUDAMessageBus &device_message_bus,
        const std::vector<device_lib::LongIndex> &message_ids,
        StepIndex step_n);

__host__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
        const CUDAMessageBus &device_message_bus,
        const std::vector<device_lib::LongIndex> &message_ids,
        StepIndex step_n);

__host__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
        const CUDAMessageBus &device_message_bus,
        const std::vector<device_lib::LongIndex> &message_ids,
        StepIndex step_n);
}