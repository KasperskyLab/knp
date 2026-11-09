/**
 * @file neuron_normalizers_test.cpp
 * @brief Tests for neuron parameters normalizers.
 * @kaspersky_support Artiom N.
 * @date 24.07.2025
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

#include <knp/framework/normalization/neuron_accessors.h>
#include <knp/framework/normalization/neuron_normalizer.h>
#include <knp/framework/population/creators.h>
#include <knp/neuron-traits/blifat.h>

#include <tests_common.h>

#include <limits>


TEST(NeuronNormalizers, NormalizePopulationPotentials)
{
    using NeuronType = knp::neuron_traits::BLIFATNeuron;
    constexpr auto neurons_count = 5;

    auto new_pop{knp::framework::population::creators::make_random<NeuronType>(neurons_count)};

    for (size_t i = 0; i < new_pop.size(); ++i)
    {
        new_pop[i].potential_ = i;
        SPDLOG_DEBUG("Old neuron potential: {}.", new_pop[i].potential_);
    }

    knp::framework::normalization::normalize_neurons<NeuronType>(
        new_pop, knp::framework::normalization::PotentialAccessor<NeuronType>(),
        knp::framework::normalization::PotentialAccessor<NeuronType>(),
        knp::framework::normalization::Rescaler<float>(0, new_pop.size() - 1, 0, 1));

    for (const auto& neuron : new_pop)
    {
        SPDLOG_DEBUG("New neuron potential: {}.", neuron.potential_);

        ASSERT_GE(neuron.potential_, 0);
        ASSERT_LE(neuron.potential_, 1);
    }
}


TEST(NeuronNormalizers, NormalizeNetworkPotentials)
{
    using NeuronType = knp::neuron_traits::BLIFATNeuron;
    constexpr auto neurons_count = 5;

    auto new_pop{knp::framework::population::creators::make_random<NeuronType>(neurons_count)};

    for (size_t i = 0; i < new_pop.size(); ++i)
    {
        new_pop[i].potential_ = i;
        SPDLOG_DEBUG("Old neuron potential: {}.", new_pop[i].potential_);
    }

    knp::framework::normalization::normalize_neurons<NeuronType>(
        new_pop, knp::framework::normalization::PotentialAccessor<NeuronType>(),
        knp::framework::normalization::PotentialAccessor<NeuronType>(),
        knp::framework::normalization::Rescaler<float>(0, new_pop.size() - 1, 0, 1));

    knp::framework::Network net;
    net.add_population(std::move(new_pop));

    for (const auto& n_pop : net.get_populations())
    {
        std::visit(
            [](const auto& pop)
            {
                using CellType = typename std::decay_t<decltype(pop)>::PopulationNeuronType;
                if constexpr (knp::core::is_neuron_v<CellType>)
                {
                    for (const auto& neuron : pop)
                    {
                        SPDLOG_DEBUG("New neuron potential: {}.", neuron.potential_);

                        ASSERT_GE(neuron.potential_, 0);
                        ASSERT_LE(neuron.potential_, 1);
                    }
                }
            },
            n_pop);
    }
}
