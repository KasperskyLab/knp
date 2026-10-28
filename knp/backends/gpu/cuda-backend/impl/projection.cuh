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
#include "delay_buffer.cuh"
#include "uid.cuh"

#include <algorithm>
#include <optional>
#include <vector>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
/**
 * @brief The CUDAProjectionBase class is a definition of a CUDA synapses.
 */
template <typename SynapseType>
struct CUDAProjectionBase
{
    /**
     * @brief Type of the projection synapses.
     */
    using ProjectionSynapseType = SynapseType;
    /**
     * @brief Projection of synapses with the specified synapse type.
     */
    using ProjectionType = CUDAProjectionBase<SynapseType>;
    /**
     * @brief Parameters of the specified synapse type.
     */
    using SynapseParameters = typename synapse_traits::synapse_parameters<SynapseType>;

    /**
     * @brief Synapse description structure that contains synapse parameters and indexes of the associated neurons.
     * @note make sure this is the same as in core projection.
     */
    using Synapse = ::cuda::std::tuple<SynapseParameters, device_lib::LongIndex, device_lib::LongIndex>;

    __host__ __device__ CUDAProjectionBase() : is_locked_(true)
     {}

    /**
     * @brief Constructor.
     * @param projection source projection.
     */
    __host__ explicit CUDAProjectionBase(const knp::core::Projection<SynapseType> &projection)
        : uid_(to_gpu_uid(projection.get_uid())),
          presynaptic_uid_(to_gpu_uid(projection.get_presynaptic())),
          postsynaptic_uid_(to_gpu_uid(projection.get_postsynaptic())),
          is_locked_(projection.is_locked())
    {
        constexpr int data_index = core::SynapseElementAccess::synapse_data;
        constexpr int source_id_index = core::SynapseElementAccess::source_neuron_id;
        constexpr int target_id_index = core::SynapseElementAccess::target_neuron_id;
        device_lib::LongIndex max_delay = 1;
        for (auto &synapse : projection)
        {
            Synapse out_synapse{std::get<data_index>(synapse), std::get<source_id_index>(synapse),
                std::get<target_id_index>(synapse)};
            max_delay = std::max(static_cast<device_lib::LongIndex>(std::get<data_index>(synapse).delay_), max_delay);
            synapses_.push_back(out_synapse);
            SPDLOG_TRACE("Synapse: weight {} delay {}", ::cuda::std::get<data_index>(out_synapse).weight_,
                         ::cuda::std::get<data_index>(out_synapse).delay_);
        }
        index_ = device_lib::build_index<SynapseType, core::source_neuron_id>(projection);
        impact_buffer_ = BitDelayBuffer{synapses_.size(), max_delay};
    }

    virtual ~CUDAProjectionBase() = default;

    __host__ __device__ void lock_weights() { is_locked_ = true; }
    __host__ __device__ void unlock_weights() { is_locked_ = false; }

    __host__ __device__ void actualize()
    {
        synapses_.actualize();
        impact_buffer_.actualize();
        index_.actualize();
    }

    /**
     * @brief Create an impact message.
     * @param current_step current step.
     * @return Synaptic impact message that would be sent.
     */
    __host__ void form_message(device_lib::LongIndex current_step);

    /**
     * @brief Get the number of synapses in the projection.
     * @return the number of contained synapses.
     */
    __host__ __device__ auto size() { return synapses_.size(); }

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
     * @brief Synapse index for quick presynaptic neuron to synapse search.
     */
    device_lib::ValueIndex index_;

    /**
     * @brief Container of synapse parameters.
     */
    cuda::device_lib::CUDAVector<Synapse> synapses_;

    /**
     * @brief Impact delay buffer.
     */
    BitDelayBuffer impact_buffer_;
    /**
     * @brief Message buffer.
     */
    SynapticImpactMessage message_buf_;
};


template <class Synapse>
struct CUDAProjection : public CUDAProjectionBase<Synapse>
{
    __host__ explicit CUDAProjection(const knp::core::Projection<Synapse> &projection)
        : CUDAProjectionBase<Synapse>(projection)
    {}

    __host__ CUDAProjection() = default;
};


template <class Synapse>
typename core::Projection<Synapse>::Synapse to_std_tuple(const typename CUDAProjection<Synapse>::Synapse &synapse)
{
    return std::make_tuple(
        ::cuda::std::get<core::synapse_data>(synapse),
        static_cast<size_t>(::cuda::std::get<core::source_neuron_id>(synapse)),
        static_cast<size_t>(::cuda::std::get<core::target_neuron_id>(synapse))
    );
}


template<class Synapse>
knp::core::Projection<Synapse> convert_to_core_projection(const CUDAProjection<Synapse> &cuda_projection)
{
    std::vector<typename CUDAProjectionBase<Synapse>::Synapse> synapses = cuda_projection.synapses_.to_std();
    auto generator = [&synapses](size_t i) -> std::optional<typename core::Projection<Synapse>::Synapse>
    {
        if (i >= synapses.size())
            return {};
        return to_std_tuple<Synapse>(synapses[i]);
    };
    knp::core::Projection<Synapse> result{to_cpu_uid(cuda_projection.uid_),
                                          to_cpu_uid(cuda_projection.presynaptic_uid_),
                                          to_cpu_uid(cuda_projection.postsynaptic_uid_),
                                          generator, synapses.size()};
    return result;
}

} // namespace knp::backends::gpu::cuda
