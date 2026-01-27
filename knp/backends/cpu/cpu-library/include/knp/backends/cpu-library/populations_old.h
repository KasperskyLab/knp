/**
 * @file populations_old.h
 * @kaspersky_support Postnikov D.
 * @date 26.01.2026
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

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>

#include <vector>

#include "populations.h"


/**
 * @brief Namespace for CPU backend's populations.
 */
namespace knp::backends::cpu
{

/**
 * @brief Find projection by type and postsynaptic uid.
 * @tparam SynapseType synapse type.
 * @tparam ProjectionContainer projection container.
 * @param projections projections.
 * @param post_uid postsynaptic uid.
 * @param exclude_locked should projections with locked weights be exluded or not.
 * @return found projections.
 */
template <class SynapseType, class ProjectionContainer>
std::vector<std::reference_wrapper<knp::core::Projection<SynapseType>>> find_projection_by_type_and_postsynaptic(
    ProjectionContainer &projections, const knp::core::UID &post_uid, bool exclude_locked)
{
    using ProjectionType = knp::core::Projection<SynapseType>;
    std::vector<std::reference_wrapper<knp::core::Projection<SynapseType>>> result;
    constexpr auto type_index = boost::mp11::mp_find<synapse_traits::AllSynapses, SynapseType>();
    for (auto &projection_wrap : projections)
    {
        if (projection_wrap.arg_.index() != type_index)
        {
            continue;
        }

        ProjectionType &projection = std::get<type_index>(projection_wrap.arg_);
        if (projection.is_locked() && exclude_locked)
        {
            continue;
        }

        if (projection.get_postsynaptic() == post_uid)
        {
            result.push_back(projection);
        }
    }
    return result;
}


/**
 * @brief Make one execution step for a population of any neurons.
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT-like parameters.
 * @param pop population to update.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @return indexes of spiked neurons.
 */
template <class Neuron>
std::optional<core::messaging::SpikeMessage> calculate_any_population(
    knp::core::Population<Neuron> &pop, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    std::vector<knp::core::messaging::SynapticImpactMessage> messages =
        endpoint.unload_messages<knp::core::messaging::SynapticImpactMessage>(pop.get_uid());
    knp::core::messaging::SpikeMessage message_out{{pop.get_uid(), step_n}, {}};
    populations::calculate_pre_impact_population_state(pop, 0, pop.size());
    populations::impact_population(pop, messages);
    populations::calculate_post_impact_population_state(pop, message_out, 0, pop.size());

    if (!message_out.neuron_indexes_.empty())
    {
        endpoint.send_message(message_out);
    }

    return message_out;
}


/**
 * @brief Make one execution step for a population of BLIFAT neurons.
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT-like parameters.
 * @param pop population to update.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @return indexes of spiked neurons.
 */
template <class BlifatLikeNeuron>
std::optional<core::messaging::SpikeMessage> calculate_blifat_population(
    knp::core::Population<BlifatLikeNeuron> &pop, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_any_population(pop, endpoint, step_n);
}


/**
 * @brief Calculate LIF population.
 * @tparam LifNeuron LIF neuron type.
 * @param pop population to calculate.
 * @param endpoint endpoint to use for message exchange.
 * @param step_n current step.
 * @return spike message with indexes of spiked neurons if population is emitting one.
 */
template <class LifNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_lif_population(
    knp::core::Population<LifNeuron> &pop, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_any_population(pop, endpoint, step_n);
}


/**
 * @brief Make one execution step for a population of `SynapticResourceSTDPNeuron` neurons.
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT-like parameters.
 * @tparam BaseSynapseType base synapse type.
 * @tparam ProjectionContainer type of a projection container.
 * @param pop population to update.
 * @param container projection container from backend.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @return message containing indexes of spiked neurons.
 */
template <class BlifatLikeNeuron, class BaseSynapseType, class ProjectionContainer>
std::optional<core::messaging::SpikeMessage> calculate_resource_stdp_population(
    knp::core::Population<neuron_traits::SynapticResourceSTDPNeuron<BlifatLikeNeuron>> &pop,
    ProjectionContainer &container, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    std::vector<knp::core::messaging::SynapticImpactMessage> messages =
        endpoint.unload_messages<knp::core::messaging::SynapticImpactMessage>(pop.get_uid());
    knp::core::messaging::SpikeMessage message_out{{pop.get_uid(), step_n}, {}};
    populations::calculate_pre_impact_population_state(pop, 0, pop.size());
    populations::impact_population(pop, messages);
    populations::calculate_post_impact_population_state(pop, message_out, 0, pop.size());

    auto working_projections = find_projection_by_type_and_postsynaptic<
        knp::synapse_traits::SynapticResourceSTDPDeltaSynapse, ProjectionContainer>(container, pop.get_uid(), true);
    cpu::populations::train_population(pop, working_projections, message_out, step_n);

    if (!message_out.neuron_indexes_.empty())
    {
        endpoint.send_message(message_out);
    }

    return message_out;
}

}  //namespace knp::backends::cpu
