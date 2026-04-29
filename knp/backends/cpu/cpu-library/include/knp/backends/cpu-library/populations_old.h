/**
 * @file populations_old.h
 * @brief Legacy interface for working with populations for backend.
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
 * @brief Namespace for CPU backend functions.
 */
namespace knp::backends::cpu
{


/**
 * @brief Find projections of a given synapse type that target a specific postsynaptic population.
 * 
 * @tparam SynapseType synapse type.
 * @tparam ProjectionContainer type of a projection container.
 *
 * @param projections container of projections to search.
 * @param post_uid UID of the postsynaptic population to match.
 * @param exclude_locked flag that determines whether projections with locked weights are omitted.
 *
 * @return vector of references to the matching projections.
 *
 * @details The function iterates over @p projections, skips entries whose variant index does not
 * match the specified synapse type, and extracts the concrete `Projection<SynapseType>`.  
 * Projections whose `is_locked()` flag is true are excluded when @p exclude_locked is set.  
 * Finally, only projections whose postsynaptic population UID matches @p post_uid are added to the result.
 *
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
 * @brief Execute one simulation step for a population of arbitrary neurons.
 *
 * @tparam Neuron type of neurons stored in the population.
 *
 * @param pop population to update.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param step_n current execution step number.
 *
 * @return spike message containing the indexes of neurons that emitted a spike during this step.
 *
 * @details The function unloads all synaptic impact messages addressed to the population from 
 * the message endpoint. Then the function calculates pre-impact state 
 * (@ref populations::calculate_pre_impact_population_state), dispatches synaptic impact messages
 * (@ref populations::impact_population), and calculates post-impact state 
 * (@ref populations::calculate_post_impact_population_state). If any spikes were generated, 
 * the functions sends a spike message back through the message endpoint.
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
 * @brief Execute one simulation step for a population of BLIFAT‑like neurons.
 *
 * @tparam BlifatLikeNeuron type of a neuron that possesses BLIFAT‑like parameters.
 *
 * @param pop population to update.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param step_n current execution step number.
 *
 * @return spike message containing the indexes of neurons that emitted a spike during this step.
 *
 * @details This function is a thin wrapper around @ref calculate_any_population. It forwards the provided population,
 *  message endpoint, and step number to the generic implementation, thereby reusing the full simulation pipeline.
 *
 */
template <class BlifatLikeNeuron>
std::optional<core::messaging::SpikeMessage> calculate_blifat_population(
    knp::core::Population<BlifatLikeNeuron> &pop, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_any_population(pop, endpoint, step_n);
}


/**
 * @brief Execute one simulation step for a population of LIF neurons.
 *
 * @tparam LifNeuron LIF neuron type.
 *
 * @param pop population to update.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param step_n current execution step number.
 *
 * @return spike message containing the indexes of neurons that emitted a spike during this step.
 *
 * @details This function simply forwards the call to @ref calculate_any_population, reusing the 
 * generic simulation pipeline.
 *
 */
template <class LifNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_lif_population(
    knp::core::Population<LifNeuron> &pop, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_any_population(pop, endpoint, step_n);
}


 /**
 * @brief Execute one simulation step for a population of @ref neuron_traits::SynapticResourceSTDPNeuron neurons.
 *
 * @tparam BlifatLikeNeuron type of a neuron with BLIFAT‑like parameters.
 * @tparam BaseSynapseType base synapse type.
 * @tparam ProjectionContainer type of a projection container.
 *
 * @param pop population to update.
 * @param container projection container supplied by the backend.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param step_n current execution step number.
 *
 * @return spike message containing the indexes of neurons that emitted a spike during this step.
 *
 * @details The function unloads all synaptic impact messages addressed to the population from 
 * the message endpoint. Then the function calculates pre-impact state 
 * (@ref populations::calculate_pre_impact_population_state), dispatches synaptic impact messages
 * (@ref populations::impact_population), and calculates post-impact state 
 * (@ref populations::calculate_post_impact_population_state). 
 * The function then retrieves projections of the @ref synapse_traits::SynapticResourceSTDPDeltaSynapse type that 
 * target the population, optionally excluding locked ones. 
 * Finally, the function trains the population with the generated spike message and sends the spike 
 * message back through the message endpoint if any neurons spiked.

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
