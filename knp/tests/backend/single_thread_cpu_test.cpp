/**
 * @file single_thread_cpu_test.cpp
 * @brief Single-threaded backend test.
 * @kaspersky_support Vartenkov An.
 * @date 07.04.2023
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

#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>
#include <tests_messaging_common.h>

#include <vector>


using Population = knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::PopulationVariants;
using Projection = knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::ProjectionVariants;


namespace knp::testing
{

class STestingBack : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    STestingBack() = default;
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

}  // namespace knp::testing


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
        knp::testing::internal::send_messages_smallest_network(in_channel_uid, endpoint, step);
        backend._step();
        auto output = std::move(knp::testing::internal::receive_messages_smallest_network(out_channel_uid, endpoint));
        if (!output.empty()) results.push_back(step);
    }

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};
    ASSERT_EQ(results, expected_results);
}


TEST(SingleThreadCpuSuite, AdditiveSTDPNetwork)
{
    using STDPDeltaProjection = knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse>;

    // Create an STDP input projection.
    auto stdp_input_projection_gen = [](size_t /*index*/) -> std::optional<STDPDeltaProjection::Synapse> {
        return STDPDeltaProjection::Synapse{{{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, {2, 2}}, 0, 0};
    };

    // Create an STDP loop projection.
    auto stdp_synapse_generator = [](size_t /*index*/) -> std::optional<STDPDeltaProjection::Synapse> {
        return STDPDeltaProjection::Synapse{{{1.0, 6, knp::synapse_traits::OutputType::EXCITATORY}, {1, 1}}, 0, 0};
    };

    auto stdp_neurons_generator = [](size_t /*index*/)  // NOLINT
        -> knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>
    { return knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>{}; };

    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    knp::testing::STestingBack backend;

    knp::core::Population<knp::neuron_traits::BLIFATNeuron> population{knp::core::UID(), stdp_neurons_generator, 1};

    auto loop_projection = STDPDeltaProjection{population.get_uid(), population.get_uid(), stdp_synapse_generator, 1};
    Projection input_projection =
        STDPDeltaProjection{knp::core::UID{false}, population.get_uid(), stdp_input_projection_gen, 1};
    knp::core::UID input_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, input_projection);

    loop_projection.get_shared_parameters().stdp_populations_[population.get_uid()] =
        STDPDeltaProjection::SharedSynapseParameters::ProcessingType::STDPAndSpike;

    backend.load_populations({population});
    backend.load_projections({input_projection, loop_projection});

    backend._init();
    auto endpoint = backend.get_message_bus().create_endpoint();

    knp::core::UID in_channel_uid, out_channel_uid;

    // Create input and output.
    backend.subscribe<knp::core::messaging::SpikeMessage>(input_uid, {in_channel_uid});
    endpoint.subscribe<knp::core::messaging::SpikeMessage>(out_channel_uid, {population.get_uid()});

    std::vector<knp::core::Step> results;

    for (knp::core::Step step = 0; step < 20; ++step)
    {
        // Send inputs on steps 0, 5, 10, 15.
        knp::testing::internal::send_messages_smallest_network(in_channel_uid, endpoint, step);
        backend._step();
        auto output = std::move(knp::testing::internal::receive_messages_smallest_network(out_channel_uid, endpoint));
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
        if (prj.get_uid() != loop_projection.get_uid())
        {
            continue;
        }

        std::transform(
            prj.begin(), prj.end(), std::back_inserter(new_synaptic_weights),
            [](const auto &synapse) { return std::get<knp::core::synapse_data>(synapse).weight_; });
    }

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};

    ASSERT_EQ(results, expected_results);
    ASSERT_NE(old_synaptic_weights, new_synaptic_weights);
}


TEST(SingleThreadCpuSuite, ResourceSTDPNetwork)
{
    using STDPDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
    using BlifatStdpPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;

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

    BlifatStdpPopulation population{
        knp::core::UID(),
        [](uint64_t) -> std::optional<BlifatStdpPopulation::NeuronParameters>
        {
            BlifatStdpPopulation::NeuronParameters neuron{{}};
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
        knp::testing::internal::send_messages_smallest_network(in_channel_uid, endpoint, step);
        backend._step();
        auto output = std::move(knp::testing::internal::receive_messages_smallest_network(out_channel_uid, endpoint));
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
