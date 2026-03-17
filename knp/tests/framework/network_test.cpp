/**
 * @file network_test.cpp
 * @brief Network class testing.
 * @kaspersky_support A. Vartenkov
 * @date 12.02.2024
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

#include <knp/framework/network.h>
#include <knp/framework/population/creators.h>

#include <tests_common.h>

#include <exception>
#include <tuple>


using BLIFATParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using Synapse = DeltaProjection::Synapse;

const auto neurons_count = 10;
const auto synapses_count = 10;


static auto create_entities()
{
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> population1(
        [=](size_t index) -> BLIFATParams
        {
            BLIFATParams params;
            params.potential_ = static_cast<double>(index / 2);
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


TEST(FrameworkSuite, EmptyNetwork)
{
    // Test default network constructor. This should create an empty network.
    knp::framework::Network network;
    // Number of projections in an empty network is zero.
    ASSERT_EQ(network.projections_count(), 0);
    // Number of populations in an empty network is zero.
    ASSERT_EQ(network.populations_count(), 0);
    // Check iterator correctness.
    ASSERT_EQ(network.begin_populations(), network.end_populations());
    ASSERT_EQ(network.begin_projections(), network.end_projections());
}


TEST(FrameworkSuite, NetworkCreation)  //!OCLINT(False positive)
{
    knp::framework::Network network;
    auto [population1, projection1] = create_entities();

    auto pop_uid = population1.get_uid();
    auto proj_uid = projection1.get_uid();

    network.add_population(std::move(population1));
    ASSERT_EQ(network.populations_count(), 1);
    ASSERT_EQ(network.projections_count(), 0);

    network.add_projection(std::move(projection1));
    ASSERT_EQ(network.populations_count(), 1);
    ASSERT_EQ(network.projections_count(), 1);

    ASSERT_TRUE(network.is_population_exists(pop_uid));
    ASSERT_EQ(network.get_population<knp::neuron_traits::BLIFATNeuron>(pop_uid).get_uid(), pop_uid);
    ASSERT_TRUE(network.is_projection_exists(proj_uid));
    ASSERT_EQ(network.get_projection<DeltaProjection::ProjectionSynapseType>(proj_uid).get_uid(), proj_uid);
    ASSERT_NE(network.get_projection<DeltaProjection::ProjectionSynapseType>(proj_uid).get_uid(), pop_uid);
}


TEST(FrameworkSuite, NetworkRemoveEntities)  //!OCLINT(False positive)
{
    knp::framework::Network network;

    auto [population1, projection1] = create_entities();

    auto [uid_pop, uid_proj] = std::make_tuple(population1.get_uid(), projection1.get_uid());

    ASSERT_EQ(network.populations_count(), 0);
    ASSERT_EQ(network.projections_count(), 0);

    network.add_population(std::move(population1));
    network.add_projection(std::move(projection1));

    ASSERT_EQ(network.populations_count(), 1);
    ASSERT_EQ(network.projections_count(), 1);

    // Incorrect UID.
    EXPECT_THROW(network.remove_population(uid_proj), std::logic_error);  //!OCLINT(False positive)
    ASSERT_EQ(network.populations_count(), 1);
    // Correct UID.
    network.remove_population(uid_pop);

    ASSERT_EQ(network.populations_count(), 0);

    // Incorrect UID.
    EXPECT_THROW(network.remove_projection(uid_pop), std::logic_error);  //!OCLINT(False positive)
    ASSERT_EQ(network.projections_count(), 1);
    // Correct UID.
    network.remove_projection(uid_proj);

    ASSERT_EQ(network.projections_count(), 0);
}


TEST(FrameworkSuite, NetworkConnectPopulations)
{
    constexpr auto src_neurons_count = 5;
    constexpr auto dst_neurons_count = 3;

    auto src_pop{knp::framework::population::creators::make_random<typename knp::neuron_traits::BLIFATNeuron>(
        src_neurons_count)};
    auto dst_pop{knp::framework::population::creators::make_random<typename knp::neuron_traits::BLIFATNeuron>(
        dst_neurons_count)};

    knp::framework::Network network;

    knp::core::UID proj_uid;

    auto connector = [&]()
    {
        proj_uid = network.connect_populations<
            typename knp::synapse_traits::DeltaSynapse, knp::neuron_traits::BLIFATNeuron,
            knp::neuron_traits::BLIFATNeuron>(src_pop, dst_pop);
        SPDLOG_DEBUG("New projection UID = {}.", std::string(proj_uid));
    };

    EXPECT_THROW(connector(), std::logic_error);  //!OCLINT(False positive)

    network.add_population(src_pop);
    EXPECT_THROW(connector(), std::logic_error);  //!OCLINT(False positive)

    SPDLOG_DEBUG("Checking projection UID = {} [False?]", std::string(proj_uid));
    ASSERT_FALSE(network.is_projection_exists(proj_uid));
    network.add_population(dst_pop);
    EXPECT_NO_THROW(connector());  //!OCLINT(False positive)

    ASSERT_EQ(network.projections_count(), 1);
    ASSERT_EQ(network.populations_count(), 2);
    SPDLOG_DEBUG("Checking projection UID = {} [True?]", std::string(proj_uid));
    ASSERT_TRUE(network.is_projection_exists(proj_uid));
    ASSERT_EQ(
        network.get_projection<knp::synapse_traits::DeltaSynapse>(proj_uid).size(),
        src_neurons_count * dst_neurons_count);
}


TEST(FrameworkSuite, NetworkConnectPopulationsArbitrary)
{
    constexpr auto src_neurons_count = 5;
    constexpr auto dst_neurons_count = 3;

    auto src_pop{
        knp::framework::population::creators::make_random<knp::neuron_traits::BLIFATNeuron>(src_neurons_count)};
    auto dst_pop{
        knp::framework::population::creators::make_random<knp::neuron_traits::BLIFATNeuron>(dst_neurons_count)};

    knp::framework::Network network;

    knp::core::UID proj_uid;

    auto connector = [&]()
    {
        proj_uid = network.connect_populations<
            knp::synapse_traits::DeltaSynapse, knp::neuron_traits::BLIFATNeuron, knp::neuron_traits::BLIFATNeuron>(
            src_pop, dst_pop,
            [dst_neurons_count](size_t index)
            {
                return std::make_tuple(
                    knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters(), index,
                    index % dst_neurons_count);
            },
            src_neurons_count);
        SPDLOG_DEBUG("New projection UID = {}.", std::string(proj_uid));
    };

    EXPECT_THROW(connector(), std::logic_error);  //!OCLINT(False positive)

    network.add_population(src_pop);
    EXPECT_THROW(connector(), std::logic_error);  //!OCLINT(False positive)

    ASSERT_FALSE(network.is_projection_exists(proj_uid));
    network.add_population(dst_pop);
    EXPECT_NO_THROW(connector());  //!OCLINT(False positive)

    ASSERT_EQ(network.projections_count(), 1);
    ASSERT_EQ(network.populations_count(), 2);
    SPDLOG_DEBUG("Checking projection UID = {} [True?]", std::string(proj_uid));
    ASSERT_TRUE(network.is_projection_exists(proj_uid));
    ASSERT_EQ(network.get_projection<knp::synapse_traits::DeltaSynapse>(proj_uid).size(), src_neurons_count);
}
