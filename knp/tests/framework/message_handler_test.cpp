/**
 * @file message_handler_test.cpp
 * @brief Model handler class testing.
 * @kaspersky_support  A. Vartenkov
 * @date 27.11.2024
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


#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/core/messaging/messaging.h>
#include <knp/framework/message_handlers.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>


TEST(MessageHandlerSuite, KWTA)
{
    std::random_device rd;
    int seed = rd();
    SPDLOG_DEBUG("Seed is {}", seed);
    knp::framework::modifier::KWtaRandomHandler kwta_handler(2, seed);
    knp::core::messaging::SpikeMessage message({{knp::core::UID{}}, {1, 2, 3, 4, 5}});
    std::vector<knp::core::messaging::SpikeMessage> msg_vec{message};
    auto out_data = kwta_handler(msg_vec);
    ASSERT_EQ(out_data.size(), 2);
    ASSERT_NE(out_data[0], out_data[1]);
    SPDLOG_DEBUG("Selected spikes are {} and {}", out_data[0], out_data[1]);

    msg_vec[0].neuron_indexes_ = {7};
    out_data = kwta_handler(msg_vec);
    ASSERT_EQ(out_data.size(), 1);
    ASSERT_EQ(out_data[0], 7);
}


TEST(MessageHandlerSuite, GroupWTASingle)
{
    std::random_device rd;
    int seed = rd();
    SPDLOG_DEBUG("Seed is {}", seed);
    // the intervals are [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, ...].
    knp::framework::modifier::GroupWtaRandomHandler group_handler({3, 6, 9}, 1, seed);

    // Message contains two spikes in group 0, one in group 1 and one in group 2, group 0 should be selected.
    knp::core::messaging::SpikeMessage message({{knp::core::UID{}}, {1, 2, 3, 6}});
    auto out_data = group_handler({message});
    // Spikes 1 and 2 are to be passed.
    ASSERT_EQ(out_data.size(), 2);
    ASSERT_EQ(out_data[0] + out_data[1], 3);

    // Message contains two spikes in group 1 and two in group 3, either group can be selected.
    message = {{knp::core::UID{}}, {1, 3, 5, 6, 9, 10}};
    out_data = group_handler({message});
    ASSERT_EQ(out_data.size(), 2);
    // Either group 1 or group 3 is selected.
    ASSERT_TRUE(out_data[0] + out_data[1] == 8 || out_data[0] + out_data[1] == 19);
    SPDLOG_DEBUG("Selected values are {} and {}.", out_data[0], out_data[1]);
}


TEST(MessageHandlerSuite, KWtaPerGroup)
{
    size_t seed = std::random_device{}();
    SPDLOG_DEBUG("Seed is {}", seed);
    // the intervals are [0, 1, 2, 3], [4, 5, 6, 7], [8, ...].
    knp::framework::modifier::KWtaPerGroup handler({4, 8}, 1, seed);

    // All neurons are activated.
    const knp::core::messaging::SpikeMessage empty_message({{knp::core::UID{}}, {}});
    knp::core::messaging::SpikeData result = handler({empty_message});
    ASSERT_EQ(result.size(), 0);
    const knp::core::messaging::SpikeMessage message({{knp::core::UID{}}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
    result = handler({message});
    ASSERT_EQ(result.size(), 3);
    ASSERT_GE(result[0], 0);
    ASSERT_LT(result[0], 4);
    ASSERT_GE(result[1], 4);
    ASSERT_LT(result[1], 8);
    ASSERT_GE(result[2], 8);
}


TEST(MessageHandlerSuite, SpikeUnionHandler)
{
    knp::framework::modifier::SpikeUnionHandler union_handler;
    knp::core::messaging::SpikeMessage message_1 = {{knp::core::UID{}, 0}, {1, 3, 5}};
    knp::core::messaging::SpikeMessage message_2 = {{knp::core::UID{}, 0}, {0, 1, 3}};
    knp::core::messaging::SpikeMessage message_3 = {{knp::core::UID{}, 0}, {3, 4, 7}};
    auto result = union_handler({message_1, message_2, message_3});
    std::sort(result.begin(), result.end());
    const decltype(result) expected_result{0, 1, 3, 4, 5, 7};
    ASSERT_EQ(result, expected_result);
}


namespace knp::testing
{

class STestingBack : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    STestingBack() = default;
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

}  // namespace knp::testing


using BlifatParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;


DeltaProjection::Synapse input_synapse_generator(size_t index)
{
    return {{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, index};
}


DeltaProjection::Synapse intermediate_synapse_generator(size_t index)
{
    return {{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, index, index};
}


TEST(MessageHandlerSuite, NetworkIntegrationTest)
{
    // In this test we are making a network that consists of:
    // - Input projection. It takes a zero input and translates it to all input neurons, instantly activating them.
    // - Input population, that consists of 6 regular BLIFAT neurons.
    // - Modifier. It divides incoming spikes into 3 groups: 0, 1; 2, 3; 4, 5 and selects only one of them.
    // - Intermediate projection that receives spikes from modifier and activates a corresponding neuron in output
    // population.
    // - Output population, still 6 neurons, but only two neighbouring neurons should ever be activated, at random.
    auto back_path = knp::testing::get_backend_path();
    constexpr int num_neurons = 6;
    // A population of 3 groups of 2 neurons per group.
    BlifatPopulation population([](size_t) { return BlifatParams{}; }, num_neurons);

    BlifatPopulation output_population([](size_t) { return BlifatParams{}; }, num_neurons);
    const knp::core::UID in_pop_uid = population.get_uid();
    const knp::core::UID out_pop_uid = output_population.get_uid();
    // A projection that activates all neurons simultaneously.
    DeltaProjection input_projection{knp::core::UID{false}, population.get_uid(), input_synapse_generator, num_neurons};

    // A projection that should later be connected to a modifier.
    DeltaProjection inter_projection{
        knp::core::UID{false}, output_population.get_uid(), intermediate_synapse_generator, num_neurons};

    const knp::core::UID input_proj_uid = input_projection.get_uid();
    const knp::core::UID inter_proj_uid = inter_projection.get_uid();

    // Create network.
    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_population(std::move(output_population));
    network.add_projection(std::move(input_projection));
    network.add_projection(std::move(inter_projection));

    // Move network to model and add input and output channels.
    knp::framework::Model model(std::move(network));
    const knp::core::UID input_uid, output_uid;
    model.add_input_channel(input_uid, input_proj_uid);
    model.add_output_channel(output_uid, out_pop_uid);

    // Generate an input spike at each step.
    auto input_gen = [](size_t step) { return knp::core::messaging::SpikeData{0}; };

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(knp::testing::get_backend_path()), {{input_uid, input_gen}});

    auto &out_channel = model_executor.get_loader().get_output_channel(output_uid);
    const std::vector<size_t> group_borders{2, 4};
    const knp::core::UID handler_uid;
    model_executor.add_spike_message_handler(
        knp::framework::modifier::GroupWtaRandomHandler{group_borders}, {in_pop_uid}, {inter_proj_uid}, handler_uid);
    constexpr int num_steps = 20;
    model_executor.start([num_steps](size_t step) { return step < num_steps; });

    const auto &spikes = out_channel.update();

    ASSERT_GE(spikes.size(), 10);
    for (const auto &msg : spikes)
    {
        ASSERT_GE(msg.header_.send_time_, 3);
        ASSERT_EQ(msg.neuron_indexes_.size(), 2);
        ASSERT_EQ(std::abs(static_cast<int>(msg.neuron_indexes_[0]) - static_cast<int>(msg.neuron_indexes_[1])), 1);
    }
}
