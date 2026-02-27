/**
 * @file prepare_network_for_inference.cpp
 * @brief Function for preparing network for inference after training.
 * @kaspersky_support D. Postnikov
 * @date 20.02.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <knp/framework/projection/creators.h>

#include <algorithm>

#include "network_functions.h"


/**
 * @brief Replace wta with projections.
 * @note We have to do this because AltAI does not support WTA.
 * @param network Annotated network.
 */
static void replace_wta_with_projections(AnnotatedNetwork& network)
{
    for (const auto& wta_data : network.data_.wta_data_)
    {
        for (const auto& sender : wta_data.first)
        {
            for (const auto& receiver : wta_data.second)
            {
                std::visit(
                    [&network, &sender, &receiver](auto& proj)
                    {
                        using ProjType = std::remove_reference_t<decltype(proj)>;
                        using SynapseType = typename ProjType::ProjectionSynapseType;
                        auto proj_copy =
                            knp::framework::projection::creators::clone_projection<SynapseType, SynapseType>(
                                proj, [&proj](size_t index) { return std::get<knp::core::synapse_data>(proj[index]); },
                                sender, proj.get_postsynaptic());
                        network.network_.remove_projection(receiver);
                        network.network_.add_projection(proj_copy);
                    },
                    network.network_.get_projection(receiver));
            }
        }
    }
    network.data_.wta_data_.clear();
}


/**
 * @brief In AltAI we need to reconstruct network for inference without WTA, and quantize weights.
 * @param backend Backend used for training.
 * @param model_desc Model description.
 * @param network Annotated network.
 */
template <>
void prepare_network_for_inference<knp::neuron_traits::AltAILIF>(
    const std::shared_ptr<knp::core::Backend>& backend, const ModelDescription& model_desc, AnnotatedNetwork& network)
{
    auto data_ranges = backend->get_network_data();
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
    network.network_ = knp::framework::Network();

    for (auto& iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto& p) { return p.get_uid(); }, population);
        if (network.data_.inference_population_uids_.find(pop_uid) != network.data_.inference_population_uids_.end())
            network.network_.add_population(std::move(population));
    }
    for (auto& iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235844
        auto projection = *iter;
        knp::core::UID proj_uid = std::visit([](const auto& p) { return p.get_uid(); }, projection);
        if (network.data_.inference_internal_projection_.find(proj_uid) !=
            network.data_.inference_internal_projection_.end())
            network.network_.add_projection(std::move(projection));
    }

    replace_wta_with_projections(network);

    for (auto proj = network.network_.begin_projections(); proj != network.network_.end_projections(); ++proj)
    {
        std::visit(
            [&network](auto&& proj)
            {
                float max_weight = 0, min_weight = 0;
                for (auto& synapse : proj)
                {
                    auto const& params = std::get<knp::core::synapse_data>(synapse);
                    if (max_weight < params.weight_) max_weight = params.weight_;
                    if (min_weight > params.weight_) min_weight = params.weight_;
                }

                const knp::core::UID post_pop_uid = proj.get_postsynaptic();
                auto& pop = std::get<knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>>(
                    network.network_.get_population(post_pop_uid));

                uint16_t max_threshold = 0;
                for (auto const& neuron : pop)
                    max_threshold =
                        std::max<uint16_t>(max_threshold, neuron.activation_threshold_ + neuron.additional_threshold_);

                const float total_max =
                    std::max({std::abs(max_weight), std::abs(min_weight), std::abs<float>(max_threshold)});
                const float scale = 255.f / total_max;

                for (auto& synapse : proj)
                {
                    auto& params = std::get<knp::core::synapse_data>(synapse);
                    params.weight_ *= scale;
                    params.weight_ = std::round(params.weight_);
                }

                for (auto& neuron : pop)
                {
                    neuron.activation_threshold_ *= scale;
                    neuron.additional_threshold_ *= scale;
                }
            },
            *proj);
    }
}
