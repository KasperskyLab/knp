/**
 * @file slicing_generators_test.cpp
 * @brief Test saving and building networks.
 * @kaspersky_support A. Vartenkov
 * @date 19.02.2026
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

#include <knp/synapse-traits/all_traits.h>
#include <knp/framework/population/neuron_generators/slicing_generators.h>
#include <knp/framework/projection/synapse_generators/synapse_slicing_generators.h>
#include <knp/core/projection.h>

#include "tests_common.h"


using ResourceProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;

TEST(SlicingGenerators, SynapseSlicing)
{
    constexpr int num_iterations = 20;
    constexpr float weight = 0.17;
    constexpr uint64_t delay = 4;
    ResourceProjection::SynapseGenerator generator =
            [](size_t step) -> std::optional<ResourceProjection::Synapse>
            {
                return std::make_tuple(ResourceProjection::SynapseParameters{
                    {weight, delay, knp::synapse_traits::OutputType::EXCITATORY}
                    }, step, step);
            };
    knp::core::UID mine_uid, from_uid, to_uid;
    ResourceProjection projection{mine_uid, from_uid, to_uid, generator, num_iterations};
    DeltaProjection sliced_projection = knp::framework::projection::synapse_generators::upcast_projection<
            knp::synapse_traits::DeltaSynapse, knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>(projection);
    ASSERT_EQ(sliced_projection.size(), num_iterations);
    auto is_correct_projection = [](const DeltaProjection &projection)
    {
        for (size_t i = 0; i < projection.size(); ++i)
        {
            const DeltaProjection::Synapse &synapse = projection[i];
            const knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse> synapse_traits_actual
                = std::get<0>(synapse);
            const knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse> synapse_traits_required
                = DeltaProjection::SynapseParameters{weight, delay,
                    knp::synapse_traits::OutputType::EXCITATORY};
            if (synapse_traits_required.weight_ != synapse_traits_actual.weight_
                || synapse_traits_required.delay_ != synapse_traits_actual.delay_)
                return false;
            if (std::get<1>(synapse) != i || std::get<2>(synapse) != i) return false;
        }
        return true;
    };
    ASSERT_TRUE(is_correct_projection(sliced_projection));
}


using BlifatParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;
using StdpBlifatParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourcePopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;

TEST(SlicingGenerators, NeuronSlicing)
{
    constexpr int num_iterations = 20;
    constexpr float potential = 11.3F;
    constexpr float threshold = 20.2F;
    ResourcePopulation::NeuronGenerator generator = [](size_t step) -> std::optional<StdpBlifatParams>
    {
        StdpBlifatParams params;
        params.potential_ = potential;
        params.activation_threshold_ = threshold + step;
        return params;
    };
    knp::core::UID uid;
    ResourcePopulation population{uid, generator, num_iterations};
    BlifatPopulation sliced_population = knp::framework::population::neurons_generators::upcast_population
            <knp::neuron_traits::BLIFATNeuron>(population);
}
