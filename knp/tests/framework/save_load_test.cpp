/**
 * @file save_load_test.cpp
 * @brief Test saving and building networks.
 * @kaspersky_support A. Vartenkov
 * @date 22.03.2024
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

#include <knp/core/projection.h>
#include <knp/framework/population/neuron_parameters_generators.h>
#include <knp/framework/sonata/network_io.h>

#include <generators.h>
#include <tests_common.h>

#include <boost/preprocessor/seq/for_each.hpp>


namespace
{

template <typename NeuronType>
knp::framework::Network make_simple_network()
{
    namespace kt = knp::testing;
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    knp::core::Population<NeuronType> population{
        knp::framework::population::neurons_generators::make_default<NeuronType>(), 1};
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> loop_projection =
        kt::DeltaProjection{population.get_uid(), population.get_uid(), kt::synapse_generator, 1};
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> input_projection =
        kt::DeltaProjection{knp::core::UID{false}, population.get_uid(), kt::input_projection_gen, 1};
    knp::framework::Network network;
    network.add_population(population);
    network.add_projection(input_projection);
    network.add_projection(loop_projection);
    return network;
}


template <class Entity>
knp::core::UID get_uid(const Entity &entity)
{
    return std::visit([](const auto &value) { return value.get_uid(); }, entity);
}


// This isn't efficient, but it doesn't matter.
template <class Content>
bool compare_container_contents(const Content &cont1, const Content &cont2, const knp::core::UID &uid)
{
    auto iter1 = std::find_if(cont1.begin(), cont1.end(), [&uid](const auto &value) { return get_uid(value) == uid; });
    auto iter2 = std::find_if(cont2.begin(), cont2.end(), [&uid](const auto &value) { return get_uid(value) == uid; });

    size_t size1 = std::visit([](auto &vec) { return vec.size(); }, *iter1);
    size_t size2 = std::visit([](auto &vec) { return vec.size(); }, *iter2);
    if (size1 != size2 || iter1->index() != iter2->index()) return false;
    // TODO: It would be nice to compare the contents, but parameter comparison is not yet implemented (1.04.24).
    return true;
}


// Compare population vectors without taking order into account.
template <class Container>
bool are_similar_containers(const Container &container_1, const Container &container_2)
{
    std::set<knp::core::UID> uids1, uids2;

    for (const auto &value : container_1) uids1.insert(get_uid(value));
    for (const auto &value : container_2) uids2.insert(get_uid(value));

    if (uids1 != uids2) return false;

    if (!std::all_of(
            uids1.begin(), uids1.end(),
            [&container_1, &container_2](const knp::core::UID &uid)
            { return compare_container_contents(container_1, container_2, uid); }))
        return false;
    return true;
}


bool are_networks_similar(const knp::framework::Network &current, const knp::framework::Network &other)
{
    if (current.get_uid() != other.get_uid()) return false;

    if (!are_similar_containers(current.get_populations(), other.get_populations())) return false;
    if (!are_similar_containers(current.get_projections(), other.get_projections())) return false;
    return true;
}

}  //namespace


class SaveLoadNetworkSuite : public ::testing::Test
{
protected:
    void TearDown() override
    {
        std::filesystem::remove_all(path_to_network_ / "network");
        std::filesystem::remove(path_to_network_ / "config.json");
    }

    std::filesystem::path path_to_network_;

    template <typename NeuronType>
    void save_load_test()
    {
        path_to_network_ = ".";
        auto network = make_simple_network<NeuronType>();
        knp::framework::sonata::save_network(network, path_to_network_);
        auto network_loaded = knp::framework::sonata::load_network(path_to_network_);
        ASSERT_TRUE(are_networks_similar(network, network_loaded));
    }
};


TEST_F(SaveLoadNetworkSuite, SaveTest)
{
    auto network = make_simple_network<knp::neuron_traits::BLIFATNeuron>();
    path_to_network_ = ".";
    knp::framework::sonata::save_network(network, path_to_network_);
    ASSERT_TRUE(std::filesystem::is_directory(path_to_network_ / "network"));
    ASSERT_TRUE(std::filesystem::is_regular_file(path_to_network_ / "network/network_config.json"));
    ASSERT_TRUE(std::filesystem::is_regular_file(path_to_network_ / "network/populations.h5"));
    ASSERT_TRUE(std::filesystem::is_regular_file(path_to_network_ / "network/projections.h5"));
    ASSERT_TRUE(std::filesystem::is_regular_file(path_to_network_ / "network/neurons.csv"));
    ASSERT_TRUE(std::filesystem::is_regular_file(path_to_network_ / "network/synapses.csv"));
}


#define NEURON_TESTS(n, _, neuron_type) save_load_test<knp::neuron_traits::neuron_type>();


TEST_F(SaveLoadNetworkSuite, SaveLoadNeuronTypes)
{
    // Testing all neuron types.
    BOOST_PP_SEQ_FOR_EACH(NEURON_TESTS, , BOOST_PP_VARIADIC_TO_SEQ(ALL_NEURONS))
}
