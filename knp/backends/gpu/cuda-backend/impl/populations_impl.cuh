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

#include "cuda_bus/message_bus.cuh"
#include "cuda_lib/vector.cuh"
#include "population.cuh"
#include <boost/mp11.hpp>
#include <cuda_runtime.h>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
using SupportedNeurons = boost::mp11::mp_list<knp::neuron_traits::BLIFATNeuron>;
using SupportedPopulations = boost::mp11::mp_transform<CUDAPopulation, SupportedNeurons>;
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
 * @brief Calculate population of BLIFAT neurons.
 * @note Population will be changed during calculation.
 * @param population population to calculate.
 * @param step current step.
 * @return set of spiked neuron indices.
 */
device_lib::CUDAVector<SpikeIndex> calculate_population(
        CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population, const CUDAMessageBus& device_message_bus,
        StepIndex step);

/**
 * @brief Calculate population of Synaptic Resource STDP Blifat neurons.
 * @param population population to calculate.
 * @param step current step.
 * @return set of spiked neuron indices.
 */
inline device_lib::CUDAVector<SpikeIndex> calculate_population(
        CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
        const CUDAMessageBus& device_message_bus,
        StepIndex step)
{
    SPDLOG_ERROR("The calculate_population function is not implemented for synaptic resource STDP BLIFAT neuron");
    return device_lib::CUDAVector<SpikeIndex>{};
}

} // namespace knp::backends::gpu::cuda