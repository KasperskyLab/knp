/**
 * @file astrocyte_population.h
 * @brief CPU calculation of astrocyte populations and synaptic weight modulation.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <knp/astrocyte-traits/all_traits.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/synapse-traits/delta.h>

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "impl/populations/astrocyte_simple/astrocyte_simple_dispatcher.h"
#include "populations.h"


/**
 * @brief Namespace for CPU backend functions.
 */
namespace knp::backends::cpu
{

/**
 * @brief Population tag containing UIDs of controlled delta-synapse projections.
 */
inline constexpr auto &astrocyte_modulated_projections_tag = astrocyte_traits::astrocyte_modulated_projections_tag;


/**
 * @brief Calculate an astrocyte population and modulate its controlled projections.
 *
 * @tparam AstrocyteType astrocyte model type.
 * @tparam ProjectionContainer type of the projection container supplied by the backend.
 *
 * @param pop astrocyte population to calculate.
 * @param container backend projection container.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param step_n current execution step number.
 *
 * @return spike message containing active astrocyte indices; its index list is empty if no astrocytes are active.
 */
template <class AstrocyteType, class ProjectionContainer>
std::optional<core::messaging::SpikeMessage> calculate_astrocyte_population(
    knp::core::Population<AstrocyteType> &pop, ProjectionContainer &container, knp::core::MessageEndpoint &endpoint,
    size_t step_n)
{
    std::vector<knp::core::messaging::SynapticImpactMessage> messages =
        endpoint.unload_messages<knp::core::messaging::SynapticImpactMessage>(pop.get_uid());
    knp::core::messaging::SpikeMessage message_out{{pop.get_uid(), step_n}, {}};

    populations::calculate_pre_impact_population_state(pop, 0, pop.size());
    populations::impact_population(pop, messages);
    populations::calculate_post_impact_population_state(pop, message_out, 0, pop.size());

    if (pop.get_tags().exists(astrocyte_modulated_projections_tag))
    {
        const auto &target_uids =
            pop.get_tags().template get_tag<std::vector<knp::core::UID>>(astrocyte_modulated_projections_tag);

        if (!pop.get_tags().exists(astrocyte_traits::astrocyte_original_weights_tag))
            pop.get_tags()[astrocyte_traits::astrocyte_original_weights_tag] =
                astrocyte_traits::AstrocyteProjectionWeights{};
        auto &original_weights = pop.get_tags().template get_tag<astrocyte_traits::AstrocyteProjectionWeights>(
            astrocyte_traits::astrocyte_original_weights_tag);

        for (size_t i = 0; i < pop.size(); ++i)
        {
            const auto &astrocyte = pop[i];
            const bool active = astrocyte.calcium_ >= astrocyte.modulation_threshold_;
            const float multiplier = active ? astrocyte.active_weight_multiplier_ : 1.F;
            const float min_weight = astrocyte.min_weight_;

            for (const auto &target_uid : target_uids)
            {
                for (auto &projection_wrap : container)
                {
                    std::visit(
                        [&target_uid, &original_weights, multiplier, min_weight](auto &projection)
                        {
                            using ProjectionType = std::decay_t<decltype(projection)>;
                            if constexpr (std::is_same_v<
                                              ProjectionType, knp::core::Projection<synapse_traits::DeltaSynapse>>)
                            {
                                if (projection.get_uid() == target_uid)
                                {
                                    auto &cached = original_weights[std::string(target_uid)];
                                    const auto &weights_tag = astrocyte_traits::astrocyte_projection_weights_tag;
                                    if (cached.empty() && projection.get_tags().exists(weights_tag))
                                        cached =
                                            projection.get_tags().template get_tag<std::vector<float>>(weights_tag);
                                    if (cached.empty())
                                    {
                                        cached.reserve(projection.size());
                                        for (auto &synapse : projection)
                                        {
                                            cached.push_back(std::get<core::synapse_data>(synapse).weight_);
                                        }
                                    }

                                    if (!projection.get_tags().exists(weights_tag))
                                        projection.get_tags()[weights_tag] = cached;
                                    size_t synapse_index = 0;
                                    for (auto &synapse : projection)
                                    {
                                        const float original_weight =
                                            synapse_index < cached.size() ? cached[synapse_index] : 0.F;
                                        std::get<core::synapse_data>(synapse).weight_ =
                                            std::max(original_weight * multiplier, min_weight);
                                        ++synapse_index;
                                    }
                                }
                            }
                        },
                        projection_wrap.arg_);
                }
            }
        }
    }

    if (!message_out.neuron_indexes_.empty())
    {
        endpoint.send_message(message_out);
    }

    return message_out;
}

}  // namespace knp::backends::cpu
