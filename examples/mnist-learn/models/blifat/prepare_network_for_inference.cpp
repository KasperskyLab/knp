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

#include "network_functions.h"


// Prepare BLIFAT network for inference execution by restoring network components.
// This template specialization prepares a trained BLIFAT network for inference operations by restoring only
// the necessary network components (populations and projections) from the training backend. Unlike AltAI which 
// requires WTA replacement and quantization, BLIFAT can use a simpler reconstruction approach that preserves the 
// original network structure while filtering for inference-specific components.
template <>
void prepare_network_for_inference<knp::neuron_traits::BLIFATNeuron>(
    const std::shared_ptr<knp::core::Backend>& backend, const ModelDescription& model_desc, AnnotatedNetwork& network)
{
    auto data_ranges = backend->get_network_data();
    
    // Clear existing network and restore only inference-relevant components.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
    network.network_ = knp::framework::Network();

    // Restore populations marked for inference use.
    for (auto& iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto& p) { return p.get_uid(); }, population);
        if (network.data_.inference_population_uids_.find(pop_uid) != network.data_.inference_population_uids_.end())
            network.network_.add_population(std::move(population));
    }

    // Restore projections marked for inference use.
    for (auto& iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235844
        auto projection = *iter;
        knp::core::UID proj_uid = std::visit([](const auto& p) { return p.get_uid(); }, projection);
        if (network.data_.inference_internal_projection_.find(proj_uid) !=
            network.data_.inference_internal_projection_.end())
            network.network_.add_projection(std::move(projection));
    }
}
