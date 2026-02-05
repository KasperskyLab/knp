/**
 * @file finalize_network.cpp
 * @brief Function for finalizing network after training.
 * @kaspersky_support D. Postnikov
 * @date 04.02.2026
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

#include "finalize_network.h"

#include <algorithm>


template <>
void finalize_network<knp::neuron_traits::AltAILIF>(AnnotatedNetwork& network, const ModelDescription& model_desc)
{
    // Quantisize weights.
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

                knp::core::UID post_pop_uid = proj.get_postsynaptic();
                auto& pop = std::get<knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>>(
                    network.network_.get_population(post_pop_uid));

                uint16_t max_threshold = 0;
                for (auto const& neuron : pop)
                    max_threshold =
                        std::max<uint16_t>(max_threshold, neuron.activation_threshold_ + neuron.additional_threshold_);

                float total_max =
                    std::max({std::abs(max_weight), std::abs(min_weight), std::abs<float>(max_threshold)});
                float scale = 255.f / total_max;
                std::cout << scale << std::endl;

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
