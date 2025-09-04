/**
 * @file blifat_population.h
 * @brief Common templates for BLIFAT population.
 * @kaspersky_support A. Vartenkov
 * @date 07.11.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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
#include <knp/backends/cpu-library/impl/blifat_population_impl.h>

/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
/**
 * @brief Make one execution step for a population of BLIFAT neurons.
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT-like parameters.
 * @param population population to update.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @return indexes of spiked neurons.
 */
template <class BlifatLikeNeuron>
std::optional<core::messaging::SpikeMessage> calculate_blifat_population(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_blifat_like_population_impl(population, endpoint, step_n);
}


/**
 * @brief Make one execution step for a population of `SynapticResourceSTDPNeuron` neurons.
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT-like parameters.
 * @tparam BaseSynapseType base synapse type.
 * @tparam ProjectionContainer type of a projection container.
 * @param population population to update.
 * @param container projection container from backend.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @return message containing indexes of spiked neurons.
 */
template <class BlifatLikeNeuron, class BaseSynapseType, class ProjectionContainer>
std::optional<core::messaging::SpikeMessage> calculate_resource_stdp_blifat_population(
    knp::core::Population<neuron_traits::SynapticResourceSTDPNeuron<BlifatLikeNeuron>> &population,
    ProjectionContainer &container, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    using StdpSynapseType = synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, BaseSynapseType>;
    auto message_opt = calculate_blifat_like_population_impl(population, endpoint, step_n);
    auto working_projections = find_projection_by_type_and_postsynaptic<StdpSynapseType, ProjectionContainer>(
        container, population.get_uid(), true);
    do_STDP_resource_plasticity(population, working_projections, message_opt, step_n);
    return message_opt;
}


/**
 * @brief Make one execution step for a population of BLIFAT neurons.
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT-like parameters.
 * @param population population to update.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @param mutex mutex.
 * @return indexes of spiked neurons.
 */
template <class BlifatLikeNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_blifat_population(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n,
    std::mutex &mutex)
{
    return calculate_blifat_like_population_impl(population, endpoint, step_n, mutex);
}

}  // namespace knp::backends::cpu
