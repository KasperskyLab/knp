/**
 * @file resource_stdp_altailif_test.cpp
 * @brief Single-threaded backend test.
 * @kaspersky_support Postnikov D.
 * @date 04.09.2025
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


TEST(SingleThreadCpuSuite, ResourceSTDPAltAILIFNetwork)
{
    using STDPDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
    using STDP_AltAILIF_Population = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>;

    // Create an STDP input projection.
    auto stdp_input_projection_gen = [](size_t /*index*/) -> std::optional<STDPDeltaProjection::Synapse>
    {
        return STDPDeltaProjection::Synapse{
            {{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, {0, 1, 2, 0.1F}}, 0, 0};
    };

    // Create an STDP loop projection.
    auto stdp_synapse_generator = [](size_t /*index*/) -> std::optional<STDPDeltaProjection::Synapse> {
        return STDPDeltaProjection::Synapse{{{1.0, 6, knp::synapse_traits::OutputType::EXCITATORY}, {0, 1, 2}}, 0, 0};
    };

    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    knp::testing::STestingBack backend;

    STDP_AltAILIF_Population population{
        knp::core::UID(),
        [](uint64_t) -> std::optional<STDP_AltAILIF_Population::NeuronParameters>
        {
            STDP_AltAILIF_Population::NeuronParameters neuron{{}};
            neuron.synaptic_resource_threshold_ = 1;
            neuron.free_synaptic_resource_ = 2;
            neuron.isi_max_ = 0;
            return neuron;
        },
        1};
    auto loop_projection = STDPDeltaProjection{population.get_uid(), population.get_uid(), stdp_synapse_generator, 1};
    Projection input_projection =
        STDPDeltaProjection{knp::core::UID{false}, population.get_uid(), stdp_input_projection_gen, 1};
    const knp::core::UID input_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, input_projection);

    backend.load_populations({population});
    backend.load_projections({input_projection, loop_projection});

    backend._init();
    backend.start_learning();
    auto endpoint = backend.get_message_bus().create_endpoint();

    const knp::core::UID in_channel_uid;
    const knp::core::UID out_channel_uid;

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
        size_t msg_count = endpoint.receive_all_messages();
        SPDLOG_DEBUG("Received {} messages.", msg_count);
        auto output = endpoint.unload_messages<knp::core::messaging::SpikeMessage>(out_channel_uid);
        SPDLOG_DEBUG("Unloaded {} messages.", output.size());
        // Write the steps on which the network sends a spike.
        if (!output.empty()) results.push_back(step);
    }

    std::vector<float> old_synaptic_weights, new_synaptic_weights;
    old_synaptic_weights.reserve(loop_projection.size());
    new_synaptic_weights.reserve(loop_projection.size());

    std::transform(
        loop_projection.begin(), loop_projection.end(), std::back_inserter(old_synaptic_weights),
        [](const auto &synapse) { return std::get<knp::core::synapse_data>(synapse).weight_; });

    for (auto proj = backend.begin_projections(); proj != backend.end_projections(); ++proj)
    {
        const auto &prj = std::get<STDPDeltaProjection>(proj->arg_);
        if (prj.get_uid() != loop_projection.get_uid()) continue;

        std::transform(
            prj.begin(), prj.end(), std::back_inserter(new_synaptic_weights),
            [](const auto &synapse) { return std::get<knp::core::synapse_data>(synapse).weight_; });
    }

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};

    ASSERT_EQ(results, expected_results);
    ASSERT_NE(old_synaptic_weights, new_synaptic_weights);
}
