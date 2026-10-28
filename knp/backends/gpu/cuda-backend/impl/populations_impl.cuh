/**
 * @file populations_impl.cuh
 * @brief Contains functions for population calculation.
 * @kaspersky_support A. Vartenkov
 * @date 17.08.2026
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

#include <knp/backends/gpu-cuda/backend.h>
#include "cuda_lib/vector.cuh"
#include "population.cuh"
#include <boost/mp11.hpp>
#include <cuda_runtime.h>

#include <vector>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
class CUDABackendImpl;

/**
 * @brief MP-based list of supported populations.
 */
using SupportedPopulations = boost::mp11::mp_transform<CUDAPopulation, CUDABackend::SupportedNeurons>;
/**
 * @brief Population variant that contains any population type specified in `SupportedPopulations`.
 * @details `PopulationVariants` takes the value of `std::variant<PopulationType_1,..., PopulationType_n>`, where
 * `PopulationType_[1..n]` is the population type specified in `SupportedPopulations`. \n
 * For example, if `SupportedPopulations` contains BLIFATNeuron and IzhikevichNeuron types,
 * then `PopulationVariants = std::variant<BLIFATNeuron, IzhikevichNeuron>`. \n
 * `PopulationVariants` retains the same order of message types as defined in `SupportedPopulations`.
 * @see ALL_NEURONS.
 */
using PopulationVariants = boost::mp11::mp_rename<SupportedPopulations, ::cuda::std::variant>;
using StepIndex = unsigned long long;

/**
 * @brief A CUDA kernel for neuron-synapse interaction.
 * @tparam Neuron the neuron type.
 * @param neurons a list of neuron parameters.
 * @param impacts synaptic impacts.
 * @param is_forcing if the message is forcing.
 */
template <class Neuron>
__global__ void calculate_neurons_impacts(
        device_lib::CUDAVectorMutableView<knp::neuron_traits::neuron_parameters<Neuron>> neurons,
        device_lib::CUDAVectorView<SynapticImpact> impacts, bool is_forcing = false);


/**
 * @brief Impact all neurons in the vector.
 * @tparam Neuron the neuron type.
 * @param neurons a vector of neuron parameters.
 * @param messages_all all synaptic impact messages.
 * @param message_ids the IDs of messages directed for the current population.
 */
template <class Neuron>
__host__ void calculate_neurons_impacts_all(
        device_lib::CUDAVectorMutableView<knp::neuron_traits::neuron_parameters<Neuron>> neurons,
        const std::vector <SynapticImpactMessage> &messages_all, std::vector <device_lib::LongIndex> message_ids)
{
    for (const auto &msg_id : message_ids)
    {
        const auto &msg = messages_all[msg_id];
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.impacts_.size());
        calculate_neurons_impacts<Neuron><<<num_blocks, num_threads>>>(neurons, msg.impacts_.view(), msg.is_forcing_);
    }
    cudaDeviceSynchronize();
}

}  // namespace knp::backends::gpu::cuda
