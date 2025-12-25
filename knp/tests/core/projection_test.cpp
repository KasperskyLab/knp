/**
 * @file projection_test.cpp
 * @brief Tests for projection entity.
 * @kaspersky_support Artiom N.
 * @date 13.04.2023
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

#include <knp/core/projection.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>

#include <cstdlib>
#include <optional>


namespace knp::testing
{

namespace knc = knp::core;

using DeltaProjection = knc::Projection<knp::synapse_traits::DeltaSynapse>;
using Synapse = DeltaProjection::Synapse;
using SynapseParameters = DeltaProjection::SynapseParameters;
using SynapseGenerator = DeltaProjection::SynapseGenerator;


SynapseGenerator make_dense_generator(std::pair<size_t, size_t> pop_sizes, SynapseParameters default_params)
{
    SynapseGenerator generator = [=](size_t index) -> std::optional<Synapse>
    {
        const uint32_t id_from = index / pop_sizes.second;
        const uint32_t id_to = index % pop_sizes.second;
        return Synapse{default_params, id_from, id_to};
    };
    return generator;
}


SynapseGenerator make_cyclic_generator(std::pair<uint32_t, uint32_t> pop_sizes, SynapseParameters default_params)
{
    SynapseGenerator generator = [=](size_t index) -> std::optional<Synapse>
    {
        const uint32_t index_from = index % pop_sizes.first;
        return Synapse{default_params, index_from, (index_from + index / pop_sizes.second) % pop_sizes.second};
    };
    return generator;
}


TEST(ProjectionSuite, Generation)  // cppcheck-suppress syntaxError
{
    // Test constructor, get_connections, get_connection, operator[], get_size.
    const size_t presynaptic_size = 99;
    const size_t postsynaptic_size = 101;
    const float weight_constant = 0.0001;

    // Create a dense projection with random weights from -0.12 to +0.17 and delay from 1 to 5.
    SynapseGenerator generator = [&](size_t iter) -> std::optional<Synapse>
    {
        const uint32_t id_from = iter / postsynaptic_size;
        const uint32_t id_to = iter % postsynaptic_size;
        float weight = weight_constant * static_cast<float>(iter);
        const SynapseParameters params{
            weight, static_cast<uint32_t>(iter / 100 + 1), knp::synapse_traits::OutputType::EXCITATORY};
        return std::optional(Synapse{params, id_from, id_to});
    };

    DeltaProjection projection(knc::UID{}, knc::UID{}, generator, presynaptic_size * postsynaptic_size);
    ASSERT_EQ(projection.size(), presynaptic_size * postsynaptic_size);
    ASSERT_EQ(std::get<knp::core::synapse_data>(projection[1000]).delay_, 11);
}


TEST(ProjectionSuite, SynapseAddition)
{
    using SynapseType = knp::synapse_traits::OutputType;
    // TODO: add_synapses, remove_presynaptic_neuron, remove_postsynaptic_neuron, disconnect.
    const uint32_t presynaptic_size = 1000;
    const uint32_t postsynaptic_size = presynaptic_size;
    // Specific neuron.
    const uint32_t neuron_index = 10;

    // Create an empty projection.
    DeltaProjection projection(knc::UID{}, knc::UID{});

    // Add synapses to an empty projection using a generator.
    SynapseGenerator generator1 = [](size_t index) {
        return Synapse{{0.0F, 1, SynapseType::EXCITATORY}, index, index};
    };
    // Add 1-to-1 synapses.
    projection.add_synapses(generator1, presynaptic_size);
    ASSERT_EQ(projection.size(), presynaptic_size);
    size_t count = projection.add_synapses(
        [neuron_index](size_t)  // NOLINT
        {
            return Synapse{
                {1.F, 2, SynapseType ::EXCITATORY},
                neuron_index,
                // Add a synapse from neuron #10 to neuron #12.
                neuron_index + 2};
        },
        1);
    ASSERT_EQ(count, 1);

    // Add synapses from presynaptic neuron `N` to postsynaptic neuron `N + 1`.
    count = projection.add_synapses(
        // Add synapses to non-empty projection using a generator.
        [&postsynaptic_size](size_t index) {
            return Synapse{{0.1F, 2, SynapseType::EXCITATORY}, index, (index + 1) % postsynaptic_size};
        },
        presynaptic_size);
    ASSERT_EQ(count, presynaptic_size);
    ASSERT_EQ(projection.size(), 2 * presynaptic_size + 1);

    // Check that neuron #10 now has three connections: #10, #11 and #12.
    std::vector<Synapse> connections;
    std::copy_if(
        projection.begin(), projection.end(), std::back_inserter(connections),
        [neuron_index](const Synapse &syn) { return std::get<knp::core::source_neuron_id>(syn) == neuron_index; });

    ASSERT_EQ(connections.size(), 3);
    for (size_t i = 0; i < 3; ++i)
    {
        // The ordering of values is implementation-defined, so we just check that all connections are present.
        ASSERT_NE(
            std::find_if(
                connections.begin(), connections.end(),
                [&i, neuron_index](const Synapse &syn)
                { return std::get<knp::core::target_neuron_id>(syn) == neuron_index + i; }),
            connections.end());
    }
}


TEST(ProjectionSuite, DeletePresynapticTest)
{
    const uint32_t size_from = 99;
    const uint32_t size_to = 101;
    const size_t synapses_per_neuron = 5;
    const size_t neuron_index = 10;
    auto generator =
        make_cyclic_generator({size_from, size_to}, {0.0F, 1, knp::synapse_traits::OutputType::EXCITATORY});
    DeltaProjection projection{knc::UID{}, knc::UID{}, generator, size_from * synapses_per_neuron};
    const size_t count = projection.remove_presynaptic_neuron_synapses(neuron_index);
    ASSERT_EQ(count, synapses_per_neuron);
    // The size of projection is correct.
    ASSERT_EQ(projection.size(), (size_from - 1) * synapses_per_neuron);
    ASSERT_EQ(
        std::find_if(
            projection.begin(), projection.end(),
            [&](const Synapse &synapse) { return std::get<knp::core::source_neuron_id>(synapse) == neuron_index; }),
        projection.end());  // Ensure that all the synapses that should have been deleted are actually deleted.
}


TEST(ProjectionSuite, DeletePostsynapticTest)
{
    const uint32_t size_from = 99;
    const uint32_t size_to = 101;
    const size_t synapses_per_neuron = 10;
    const size_t neuron_index = 15;
    auto generator =
        make_cyclic_generator({size_from, size_to}, {0.0F, 1, knp::synapse_traits::OutputType::EXCITATORY});
    DeltaProjection projection{knc::UID{}, knc::UID{}, generator, size_from * synapses_per_neuron};
    const size_t count = projection.remove_postsynaptic_neuron_synapses(neuron_index);
    ASSERT_EQ(count, synapses_per_neuron);
    ASSERT_EQ(projection.size(), (size_from - 1) * synapses_per_neuron);  // The size of projection is correct.
    ASSERT_EQ(
        std::find_if(
            projection.begin(), projection.end(),
            [&](const Synapse &synapse) { return std::get<knp::core::target_neuron_id>(synapse) == neuron_index; }),
        projection.end());  // Ensure that all the synapses that should have been deleted are actually deleted.
}


TEST(ProjectionSuite, SynapseRemoval)
{
    // Projection does not contain the neuron we are removing.
    DeltaProjection projection{knc::UID{}, knc::UID{}};
    size_t count = projection.remove_presynaptic_neuron_synapses(100);
    ASSERT_EQ(count, 0);
    ASSERT_EQ(projection.size(), 0);

    const uint32_t presynaptic_size = 100;
    const uint32_t postsynaptic_size = presynaptic_size;
    const size_t synapses_per_neuron = 4;
    const size_t total_connections = presynaptic_size * synapses_per_neuron;

    // If we run this generator `N * presynaptic size` times, we will have 1 to N cycled connections of `x -> x, x -> x
    // + 1, ... x -> x + N`.
    count = projection.add_synapses(
        make_cyclic_generator(
            {presynaptic_size, postsynaptic_size}, {0, 1, knp::synapse_traits::OutputType::EXCITATORY}),
        total_connections);
    ASSERT_EQ(count, total_connections);

    // Delete a single synapse.
    projection.remove_synapse(0);
    // A synapse was deleted.
    ASSERT_EQ(projection.size(), total_connections - 1);
    // The deleted synapse is the correct one.
    ASSERT_EQ(std::get<knp::core::source_neuron_id>(projection[0]), 1);

    // Delete all synapses.
    projection.clear();
    ASSERT_EQ(projection.size(), 0);
}


TEST(ProjectionSuite, LockTest)
{
    DeltaProjection projection(knc::UID{}, knc::UID{});
    ASSERT_TRUE(projection.is_locked());
    projection.unlock_weights();
    ASSERT_FALSE(projection.is_locked());
    projection.lock_weights();
    ASSERT_TRUE(projection.is_locked());
}


TEST(ProjectionSuite, DisconnectNeurons)
{
    const uint32_t presynaptic_size = 9;
    const uint32_t postsynaptic_size = 11;
    auto generator = make_dense_generator(
        {presynaptic_size, postsynaptic_size}, {0.0, 1, knp::synapse_traits::OutputType::EXCITATORY});
    DeltaProjection projection{knc::UID{}, knc::UID{}, generator, presynaptic_size * postsynaptic_size};
    const size_t count = projection.remove_synapse_if(
        [](const Synapse &synapse)
        {
            return (std::get<knp::core::source_neuron_id>(synapse) == 0) &&
                   (std::get<knp::core::target_neuron_id>(synapse) == 1);
        });

    ASSERT_EQ(count, 1);
    ASSERT_EQ(
        std::count_if(
            projection.begin(), projection.end(),
            [](const Synapse &synapse) { return std::get<knp::core::source_neuron_id>(synapse) == 0; }),
        postsynaptic_size - 1);
    ASSERT_EQ(
        std::count_if(
            projection.begin(), projection.end(),
            [](const Synapse &synapse) { return std::get<knp::core::target_neuron_id>(synapse) == 1; }),
        presynaptic_size - 1);
}


TEST(ProjectionSuite, GetUIDTest)
{
    const knc::UID uid_from(true);
    const knc::UID uid_to(true);

    ASSERT_NE(uid_from, uid_to);
    auto generator = make_dense_generator({10, 10}, {0, 1, knp::synapse_traits::OutputType::EXCITATORY});
    const DeltaProjection projection(uid_from, uid_to, generator, 100);
    ASSERT_EQ(projection.get_presynaptic(), uid_from);
    ASSERT_EQ(projection.get_postsynaptic(), uid_to);
}

}  // namespace knp::testing
