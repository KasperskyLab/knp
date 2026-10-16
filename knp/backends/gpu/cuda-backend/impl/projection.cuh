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
#include "cuda_lib/value_index.cuh"
#include "cuda_bus/synaptic_impact_message.cuh"
#include "uid.cuh"


/**
 * @brief Namespace for CUDA backend.
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
     * @note make sure this is the same as in core projection.
     */
    using Synapse = ::cuda::std::tuple<SynapseParameters, device_lib::LongIndex, device_lib::LongIndex>;

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
        constexpr int data_index = core::SynapseElementAccess::synapse_data;
        constexpr int source_id_index = core::SynapseElementAccess::source_neuron_id;
        constexpr int target_id_index = core::SynapseElementAccess::target_neuron_id;
        for (auto &synapse : projection)
        {
            Synapse out_synapse{std::get<data_index>(synapse), std::get<source_id_index>(synapse),
                    std::get<target_id_index>(synapse)};
            synapses_.push_back(out_synapse);
            SPDLOG_TRACE("Synapse: weight {} delay {}", ::cuda::std::get<data_index>(out_synapse).weight_,
                         ::cuda::std::get<data_index>(out_synapse).delay_);
        }
        index_ = device_lib::build_index<SynapseType>(projection);
    }

    __host__ __device__ void lock_weights() { is_locked_ = true; }
    __host__ __device__ void unlock_weights() { is_locked_ = false; }

    __host__ __device__ void actualize()
    {
        synapses_.actualize();
        impact_indexes_.actualize();
        sending_steps_.actualize();
        index_.actualize();
    }

    /**
     * @brief Add new impacts to the impact queue.
     * @param new_impacts_indexes indexes of the synapses that will emit impacts.
     * @param new_sending_steps the target step when the impact will be emitted.
     * @note pre-sort new_impacts_indexes by new_sending_steps.
     */
    __host__ void add_impacts(const device_lib::CUDAVector<device_lib::LongIndex> &new_impacts_indexes,
                              const device_lib::CUDAVector<device_lib::LongIndex> &new_sending_steps)
    {
        assert(new_impacts_indexes.size() == new_sending_steps.size());
        if (new_impacts_indexes.size() == 0) return;
        const auto size = new_impacts_indexes.size();
        const auto out_size = new_impacts_indexes.size() + impact_indexes_.size();
        device_lib::LongIndex *res_steps;
        device_lib::LongIndex *res_impacts;
        cudaMalloc(&res_steps, sizeof(device_lib::LongIndex) * out_size);
        cudaMalloc(&res_impacts, sizeof(device_lib::LongIndex) * out_size);
        thrust::merge_by_key(thrust::device, new_sending_steps.data(), new_sending_steps.data() + size,
                             sending_steps_.data(), sending_steps_.data() + sending_steps_.size(),
                             new_impacts_indexes.data(), impact_indexes_.data(), res_steps, res_impacts);
        impact_indexes_ = device_lib::CUDAVector<device_lib::LongIndex>{res_impacts, out_size};
        sending_steps_ = device_lib::CUDAVector<device_lib::LongIndex>{res_steps, out_size};
    }

    /**
     * @brief Create an impact message.
     * @param current_step current step.
     * @return Synaptic impact message that would be sent.
     */
    __host__ void form_message(device_lib::LongIndex current_step);

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
     * @brief Synapse index for quick neuron to synapse search.
     */
    device_lib::ValueIndex index_;

    /**
     * @brief Container of synapse parameters.
     */
    cuda::device_lib::CUDAVector<Synapse> synapses_;

    /**
     * @brief Incoming impacts for the projection.
     */
    device_lib::CUDAVector<device_lib::LongIndex> impact_indexes_;
    device_lib::CUDAVector<device_lib::LongIndex> sending_steps_;

    /**
     * @brief Message buffer.
     */
    SynapticImpactMessage message_buf_;
};

} // namespace knp::backends::gpu::cuda
