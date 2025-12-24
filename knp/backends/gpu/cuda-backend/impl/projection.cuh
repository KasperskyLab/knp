/**
 * @file projection.cuh
 * @brief GPU projection implementation.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
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

#include <tuple>
#include <utility>

#include <knp/core/projection.h>
#include <knp/synapse-traits/all_traits.h>

#include "cuda_lib/vector.cuh"
#include "cuda_bus/synaptic_impact_message.cuh"
#include "uid.cuh"


/**
 * @brief Namespace for single-threaded backend.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief The CUDAProjection class is a definition of a CUDA synapses.
 */
template <typename SynapseType>
struct CUDAProjection
{
    /**
     * @brief Type of the projection synapses.
     */
    using ProjectionSynapseType = SynapseType;
    /**
     * @brief Projection of synapses with the specified synapse type.
     */
    using ProjectionType = CUDAProjection<SynapseType>;
    /**
     * @brief Parameters of the specified synapse type.
     */
    using SynapseParameters = typename synapse_traits::synapse_parameters<SynapseType>;

    /**
     * @brief Synapse description structure that contains synapse parameters and indexes of the associated neurons.
     */
    using Synapse = ::cuda::std::tuple<SynapseParameters, uint32_t, uint32_t>;

    __host__ __device__ CUDAProjection()
    #if !defined(__CUDA_ARCH__)
             :  is_locked_(true)
    #endif
    {}

    /**
     * @brief Constructor.
     * @param projection source projection.
     */
    __host__ explicit CUDAProjection(const knp::core::Projection<SynapseType> &projection)
        : uid_(to_gpu_uid(projection.get_uid())),
          presynaptic_uid_(to_gpu_uid(projection.get_presynaptic())),
          postsynaptic_uid_(to_gpu_uid(projection.get_postsynaptic())),
          is_locked_(projection.is_locked())
    {
        synapses_.reserve(projection.size());
        for (auto &synapse : projection)
        {
            Synapse out_synapse{std::get<0>(synapse), std::get<1>(synapse), std::get<2>(synapse)};
            synapses_.push_back(out_synapse);
            // TODO TEMP
//            SPDLOG_TRACE("Synapse: weight {} delay {}", ::cuda::std::get<0>(synapses_.copy_at(0).weight_,
//                                                                            synapses_.copy_at(0).delay_));
            SPDLOG_TRACE("Synapse: weight {} delay {}", ::cuda::std::get<0>(out_synapse).weight_,
                         ::cuda::std::get<0>(out_synapse).delay_);
            Synapse syn_copy = synapses_.copy_at(synapses_.size() - 1);
            SPDLOG_TRACE("Synapse2: weight {} delay {}", ::cuda::std::get<0>(syn_copy).weight_,
                         ::cuda::std::get<0>(syn_copy).delay_);
        }
    }

    __host__ __device__ void lock_weights() { is_locked_ = true; }
    __host__ __device__ void unlock_weights() { is_locked_ = false; }

    __host__ __device__ void actualize()
    {
        synapses_.actualize();
        messages_.actualize();
    };

    /**
     * @brief UID.
     */
    cuda::UID uid_;

    /**
     * @brief UID of the population that sends spikes to the projection (presynaptic population)
     */
    cuda::UID presynaptic_uid_;

    /**
     * @brief UID of the population that receives synapse responses from this projection (postsynaptic population).
     */
    cuda::UID postsynaptic_uid_;

    /**
     * @brief Return `false` if the weight change for synapses is not locked.
     */
    bool is_locked_;

    /**
     * @brief Container of synapse parameters.
     */
    cuda::device_lib::CUDAVector<Synapse> synapses_;

    /**
     * @brief Messages container.
     */
    // cppcheck-suppress unusedStructMember
    device_lib::CUDAVector<cuda::SynapticImpactMessage> messages_;
};

} // namespace knp::backends::gpu::cuda
