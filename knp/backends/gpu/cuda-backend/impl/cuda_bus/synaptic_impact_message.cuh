/**
 * @file synaptic_impact_message.cuh
 * @brief Synaptic impact message class for CUDA
 * @kaspersky_support Artiom N.
 * @date 04.04.2025
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
#include <knp/core/messaging/synaptic_impact_message.h>
#include <knp/synapse-traits/output_types.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include "../cuda_lib/vector.cuh"
#include "../uid.cuh"
#include "message_header.cuh"


/**
 * @brief Messaging namespace.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief Structure that contains the synaptic impact value and indexes of presynaptic and postsynaptic neurons.
 * @details Synaptic impact changes parameters of neurons after the synapses state is calculated.
 */
class SynapticImpact
{
public:
    /**
     * @brief Index of the population synapse.
     */
    uint64_t connection_index_;

    /**
     * @brief Value used to change neuron membrane potential.
     */
    float impact_value_;

    /**
     * @brief Synapse type that might define the value role inside the neuron function.
     */
    knp::synapse_traits::OutputType synapse_type_;

    /**
     * @brief Index of the presynaptic neuron connected to the synapse.
     */
    uint32_t presynaptic_neuron_index_;

    /**
     * @brief Index of the postsynaptic neuron connected to the synapse.
     */
    uint32_t postsynaptic_neuron_index_;

    /**
     * @brief Compare synaptic impact messages.
     * @return `true` if synaptic impacts are equal.
     */
    __host__ __device__ bool operator==(const SynapticImpact &other) const
    {
        return connection_index_ == other.connection_index_ && impact_value_ == other.impact_value_
                && synapse_type_ == other.synapse_type_ && presynaptic_neuron_index_ == other.presynaptic_neuron_index_
                && postsynaptic_neuron_index_ == other.postsynaptic_neuron_index_;
    }

    __host__ __device__ bool operator!=(const SynapticImpact &other) const
    {
        return !(*this == other);
    }
};


/**
 * @brief Structure of the synaptic impact message.
 */
class SynapticImpactMessage
{
public:
    /**
     * @brief Message header.
     */
    MessageHeader header_;

    /**
     * @brief UID of the population that sends spikes to the projection.
     */
    cuda::UID presynaptic_population_uid_;

    /**
     * @brief UID of the population that receives impacts from the projection.
     */
    cuda::UID postsynaptic_population_uid_;

    /**
     * @brief Synaptic impacts.
     */
    device_lib::CUDAVector<SynapticImpact> impacts_;

    /**
     * @brief Boolean value that defines whether the signal is from a projection without plasticity.
     * @details The parameter is used in training. Use `true` if the signal is from a projection without plasticity.
     * @todo Try to remove this when fixing main; this parameter is too specific to be a part of a general message.
     */
    bool is_forcing_ = false;

    __host__ __device__ void actualize() { impacts_.actualize(); }


    __host__ __device__ bool operator==(const SynapticImpactMessage &other)
    {
        return header_ == other.header_
            && presynaptic_population_uid_ == other.presynaptic_population_uid_
            && postsynaptic_population_uid_ == other.postsynaptic_population_uid_
            && impacts_ == other.impacts_;
    }

    __host__ __device__ bool operator!=(const SynapticImpactMessage &other) { return !(*this == other); }
};


/**
 * @brief Check if two synaptic impact messages are the same.
 * @param sm1 first message.
 * @param sm2 second message.
 * @return `true` if both messages are the same.
 */
bool operator==(const SynapticImpactMessage &sm1, const SynapticImpactMessage &sm2);


namespace detail
{
inline __host__ __device__ cuda::SynapticImpact make_gpu_impact(const knp::core::messaging::SynapticImpact &host_impact)
{
    return {.connection_index_ = host_impact.connection_index_,
            .impact_value_ = host_impact.impact_value_,
            .synapse_type_ = host_impact.synapse_type_,
            .presynaptic_neuron_index_ = host_impact.presynaptic_neuron_index_,
            .postsynaptic_neuron_index_ = host_impact.postsynaptic_neuron_index_ };
}

inline __host__ __device__  knp::core::messaging::SynapticImpact make_host_impact(
        const cuda::SynapticImpact &device_impact)
{
    return {.connection_index_ = device_impact.connection_index_,
            .impact_value_ = device_impact.impact_value_,
            .synapse_type_ = device_impact.synapse_type_,
            .presynaptic_neuron_index_ = device_impact.presynaptic_neuron_index_,
            .postsynaptic_neuron_index_ = device_impact.postsynaptic_neuron_index_ };
}


cuda::SynapticImpactMessage make_gpu_message(const knp::core::messaging::SynapticImpactMessage &host_message);

/**
 * @brief make host message from a GPU pointer to GPU message.
 * @param gpu_message GPU pointer to GPU message.
 * @return host message with the same data.
 */
knp::core::messaging::SynapticImpactMessage make_host_message(const cuda::SynapticImpactMessage *gpu_message);

}  // namespace detail
}  // namespace knp::backends::gpu::cuda
