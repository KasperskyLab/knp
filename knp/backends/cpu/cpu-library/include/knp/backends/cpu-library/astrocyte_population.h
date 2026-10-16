#pragma once

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/neuron-traits/astrocyte.h>
#include <knp/synapse-traits/delta.h>

#include <algorithm>
#include <map>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

#include "populations.h"
#include "impl/populations/astrocyte/astrocyte_dispatcher.h"


/**
 * @brief Namespace for CPU backend functions.
 */
namespace knp::backends::cpu
{

/**
 * @brief Name of the population tag that stores the UIDs of the delta-synapse projections that an astrocyte
 * population controls.
 */
constexpr auto &astrocyte_modulated_projections_tag = neuron_traits::astrocyte_modulated_projections_tag;


/**
 * @brief Execute one simulation step for a population of astrocytes, and apply the resulting weight
 * modulation to the delta-synapse projections it controls.
 *
 * @tparam ProjectionContainer type of the projection container supplied by the backend.
 *
 * @param pop astrocyte population to calculate.
 * @param container live projection container owned by the backend.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param step_n current execution step number.
 *
 * @return spike message with the astrocyte's index whenever it is active; empty otherwise.
 */
template <class ProjectionContainer>
std::optional<core::messaging::SpikeMessage> calculate_astrocyte_population(
    knp::core::Population<neuron_traits::AstrocyteCell> &pop, ProjectionContainer &container,
    knp::core::MessageEndpoint &endpoint, size_t step_n)
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

        // Original weight of every synapse in every controlled projection, cached on first use.
        static std::map<knp::core::UID, std::vector<float>> original_weights;

        for (size_t i = 0; i < pop.size(); ++i)
        {
            const auto &neuron = pop[i];
            const bool active = neuron.calcium_ > neuron.modulation_threshold_;
            const float multiplier = active ? neuron.active_weight_multiplier_ : 1.F;
            const float min_weight = neuron.min_weight_;

            for (const auto &target_uid : target_uids)
            {
                for (auto &projection_wrap : container)
                {
                    std::visit(
                        [&target_uid, multiplier, min_weight](auto &projection)
                        {
                            using ProjectionType = std::decay_t<decltype(projection)>;
                            if constexpr (std::is_same_v<
                                              ProjectionType, knp::core::Projection<synapse_traits::DeltaSynapse>>)
                            {
                                if (projection.get_uid() == target_uid)
                                {
                                    auto &cached = original_weights[target_uid];
                                    if (cached.empty())
                                    {
                                        cached.reserve(projection.size());
                                        for (auto &synapse : projection)
                                        {
                                            cached.push_back(std::get<core::synapse_data>(synapse).weight_);
                                        }
                                    }

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
