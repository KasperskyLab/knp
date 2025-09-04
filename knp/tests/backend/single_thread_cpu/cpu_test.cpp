/**
 * @file single_thread_cpu_test.cpp
 * @brief Single-threaded backend test.
 * @kaspersky_support Vartenkov An.
 * @date 07.04.2023
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

#include <backend_single_thread_cpu_common.h>
#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>


TEST(SingleThreadCpuSuite, SmallestNetwork)
{
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    knp::testing::STestingBack backend;

    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};
    Projection loop_projection =
        knp::testing::DeltaProjection{population.get_uid(), population.get_uid(), knp::testing::synapse_generator, 1};
    Projection input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};
    knp::core::UID const input_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, input_projection);

    backend.load_populations({population});
    backend.load_projections({input_projection, loop_projection});

    backend._init();
    auto endpoint = backend.get_message_bus().create_endpoint();

    const knp::core::UID in_channel_uid, out_channel_uid;

    // Create input and output.
    backend.subscribe<knp::core::messaging::SpikeMessage>(input_uid, {in_channel_uid});
    endpoint.subscribe<knp::core::messaging::SpikeMessage>(out_channel_uid, {population.get_uid()});

    std::vector<knp::core::Step> results;

    for (knp::core::Step step = 0; step < 20; ++step)
    {
        // Send inputs on steps 0, 5, 10, 15.
        if (step % 5 == 0)
        {
            knp::core::messaging::SpikeMessage message{{in_channel_uid, step}, {0}};
            endpoint.send_message(message);
        }
        backend._step();
        endpoint.receive_all_messages();
        // Write the steps on which the network sends a spike.
        if (!endpoint.unload_messages<knp::core::messaging::SpikeMessage>(out_channel_uid).empty())
        {
            results.push_back(step);
        }
    }

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};
    ASSERT_EQ(results, expected_results);
}


TEST(SingleThreadCpuSuite, NeuronsGettingTest)
{
    const knp::testing::STestingBack backend;

    auto s_neurons = backend.get_supported_neurons();

    ASSERT_LE(s_neurons.size(), boost::mp11::mp_size<knp::neuron_traits::AllNeurons>());
    ASSERT_EQ(s_neurons[0], "BLIFATNeuron");
}


TEST(SingleThreadCpuSuite, SynapsesGettingTest)
{
    const knp::testing::STestingBack backend;

    auto s_synapses = backend.get_supported_synapses();

    ASSERT_LE(s_synapses.size(), boost::mp11::mp_size<knp::synapse_traits::AllSynapses>());
    ASSERT_EQ(s_synapses[0], "DeltaSynapse");
}
