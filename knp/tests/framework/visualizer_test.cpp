/**
 * @file visualizer_test.cpp
 * @brief Network visualizer tests.
 * @kaspersky_support Artiom N.
 * @date 17.03.2023
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


#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <tests_common.h>


using BLIFATParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using Synapse = DeltaProjection::Synapse;

const auto neurons_count = 10;
const auto synapses_count = 20;


auto create_entities()
{
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> population1(
        [=](size_t index) -> BLIFATParams
        {
            BLIFATParams params;
            params.potential_ = static_cast<double>(index);
            return params;
        },
        neurons_count);

    DeltaProjection projection1(
        knp::core::UID{}, knp::core::UID{},
        [=](size_t index) -> std::optional<Synapse>
        {
            const uint32_t id_from = index;
            const uint32_t id_to = index;
            return Synapse{{}, id_from, id_to};
        },
        synapses_count);

    return std::make_tuple(population1, projection1);
}


TEST(FrameworkSuite, VisualizerTest)
{
    knp::framework::Network network;
    auto [population1, projection1] = create_entities();

    auto pop_uid = population1.get_uid();
    auto proj_uid = projection1.get_uid();

    const knp::framework::NetworkGraph net_graph(network);

    knp::framework::print_network_description(net_graph);
    knp::framework::position_network_test(
        knp::framework::NetworkGraph(network), knp::framework::divide_graph_by_connectivity(net_graph)[0], {1, 2});
}
