/**
 * @file population_test.cpp
 * @brief Tests for population entity.
 * @kaspersky_support Artiom N.
 * @date 13.02.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <knp/core/population.h>
#include <knp/neuron-traits/blifat.h>

#include <tests_common.h>


namespace knp::testing
{

using BLIFATParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;

BLIFATParams neuron_generator(size_t index)
{
    BLIFATParams params;
    params.potential_ = static_cast<double>(index);
    return params;
}


const auto neurons_count = 10;


TEST(PopulationSuite, Create)  // cppcheck-suppress syntaxError
{
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> population(neuron_generator, neurons_count);

    ASSERT_EQ(neurons_count, population.size());

    size_t n_counter = 0;
    for (const auto &neuron : population)
    {
        ASSERT_EQ(n_counter++, neuron.potential_);
    }
}


TEST(PopulationSuite, RemoveNeurons)
{
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> population(neuron_generator, neurons_count);

    ASSERT_EQ(neurons_count, population.size());

    const std::vector<size_t> indexes_to_remove{1, 3, 5};

    assert(indexes_to_remove.size() < neurons_count);

    const auto prev_size = population.size();
    population.remove_neurons(indexes_to_remove);

    ASSERT_EQ(population.size(), prev_size - indexes_to_remove.size());
}


TEST(PopulationSuite, SetNeuronParameter)
{
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> population(neuron_generator, neurons_count);

    ASSERT_EQ(neurons_count, population.size());

    constexpr size_t p_index = neurons_count / 2;

    population[p_index].potential_ = 150;

    ASSERT_EQ(150, population[p_index].potential_);
}

}  // namespace knp::testing
