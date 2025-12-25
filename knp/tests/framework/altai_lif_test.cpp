/**
 * @file backend_loader_test.cpp
 * @brief Backend loading testing.
 * @kaspersky_support Artiom N.
 * @date 17.03.2023
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
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/altai_lif.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>

#include <algorithm>


using Synapse = knp::synapse_traits::DeltaSynapse;


namespace knp::testing
{

class TestingBackendST : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    TestingBackendST() = default;
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

using Population = knp::core::Population<knp::neuron_traits::AltAILIF>;
using Projection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;

}  // namespace knp::testing


struct NeuronLog
{
    std::vector<int16_t> potential_;
    std::vector<size_t> spikes_;
};


constexpr int num_steps = 10;


NeuronLog run_altai_neuron(
    const knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF> &neuron, size_t steps,
    const std::vector<float> &impacts = {}, const uint32_t num_neurons = 1, const uint32_t neuron_index = 0)
{
    assert(num_neurons > neuron_index);
    const knp::core::UID pop_uid, in_uid, out_uid;
    knp::core::Population<knp::neuron_traits::AltAILIF> population{
        pop_uid, [&neuron](size_t) { return neuron; }, num_neurons};
    knp::testing::TestingBackendST backend;
    backend.subscribe<knp::core::messaging::SynapticImpactMessage>(pop_uid, {in_uid});
    auto endpoint = backend.get_message_bus().create_endpoint();
    endpoint.subscribe<knp::core::messaging::SpikeMessage>(out_uid, {pop_uid});

    backend.load_populations({population});
    backend._init();
    auto &pop = *backend.begin_populations();
    NeuronLog result;
    auto &neuron_ref = std::get<knp::core::Population<knp::neuron_traits::AltAILIF>>(pop)[neuron_index];
    for (size_t step = 0; step < steps; ++step)
    {
        const knp::core::messaging::MessageHeader header{in_uid, step};
        if (step < impacts.size())
        {
            knp::core::messaging::SynapticImpact impact{
                0, impacts[step], knp::synapse_traits::OutputType::EXCITATORY, 0, neuron_index};
            const knp::core::messaging::SynapticImpactMessage msg{
                header, knp::core::UID{false}, pop_uid, true, {impact}};
            endpoint.send_message(msg);
        }
        result.potential_.push_back(static_cast<int16_t>(neuron_ref.potential_));
        backend._step();
        endpoint.receive_all_messages();
        auto out_msgs = endpoint.unload_messages<knp::core::messaging::SpikeMessage>(out_uid);
        if (!out_msgs.empty() && !out_msgs[0].neuron_indexes_.empty())
        {
            if (std::find(out_msgs[0].neuron_indexes_.begin(), out_msgs[0].neuron_indexes_.end(), neuron_index) !=
                out_msgs[0].neuron_indexes_.end())
                result.spikes_.push_back(step);
        }
    }
    return result;
}


TEST(AltAiSuite, NeuronPotentialLeakRev)
{
    constexpr int starting_potential = 100;
    constexpr int potential_leak = -25;
    constexpr int neuron_num = 3;
    constexpr int neuron_ind = 2;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    base_neuron.activation_threshold_ = 2 * starting_potential;  // We don't want activations in this test.
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.potential_ = starting_potential;
    base_neuron.leak_rev_ = true;

    auto results = run_altai_neuron(base_neuron, num_steps, {}, neuron_num, neuron_ind);
    const std::vector<int16_t> expected_results = {100, 75, 50, 25, 0, -25, 0, -25, 0, -25};
    ASSERT_EQ(results.potential_, expected_results);
    ASSERT_TRUE(results.spikes_.empty());
}


TEST(AltAiSuite, NeuronPotentialLeakNoRev)
{
    constexpr int starting_potential = 100;
    constexpr int potential_leak = -25;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    // This is a leak test, so no activations right now, negative or otherwise.
    base_neuron.activation_threshold_ = 2 * starting_potential;
    base_neuron.negative_activation_threshold_ = -2 * starting_potential;
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.potential_ = starting_potential;
    base_neuron.leak_rev_ = false;

    auto result = run_altai_neuron(base_neuron, num_steps);
    const std::vector<int16_t> expected_results = {100, 75, 50, 25, 0, -25, -50, -75, -100, -125};
    ASSERT_EQ(result.potential_, expected_results);
    ASSERT_TRUE(result.spikes_.empty());
}


TEST(AltAiSuite, SaturateTest)
{
    constexpr int starting_potential = 100;
    constexpr int potential_leak = -50;
    constexpr uint16_t neg_threshold = 60;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    base_neuron.potential_ = starting_potential;
    base_neuron.activation_threshold_ = 2 * starting_potential;
    base_neuron.negative_activation_threshold_ = neg_threshold;
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.leak_rev_ = false;
    base_neuron.saturate_ = true;

    auto result = run_altai_neuron(base_neuron, num_steps);
    const std::vector<int16_t> expected_results = {100, 50, 0, -50, -60, -60, -60, -60, -60, -60};
    ASSERT_EQ(result.potential_, expected_results);
    ASSERT_TRUE(result.spikes_.empty());
}


TEST(AltAiSuite, SpikeGenerationTest)
{
    constexpr int potential_leak = 50;
    constexpr int threshold = 110;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.activation_threshold_ = threshold;
    auto result = run_altai_neuron(base_neuron, num_steps);
    const std::vector<int16_t> expected_potential{0, 50, 100, 0, 50, 100, 0, 50, 100, 0};
    const std::vector<size_t> expected_spikes{2, 5, 8};
    ASSERT_EQ(result.potential_, expected_potential);
    ASSERT_EQ(result.spikes_, expected_spikes);
}


TEST(AltAiSuite, SingleNeuronImpactTest)
{
    constexpr int potential_leak = -30;
    constexpr int threshold = 100;
    constexpr int num_neurons = 5;
    constexpr int neuron_index = 3;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.activation_threshold_ = threshold;
    const std::vector<float> impacts{130, 50, 50, 50, 50, 0, 100, 80, 80, 80};
    auto result = run_altai_neuron(base_neuron, num_steps, impacts, num_neurons, neuron_index);
    const std::vector<int16_t> expected_potential{0, 0, 20, 40, 60, 80, 50, 0, 50, 0};
    const std::vector<size_t> expected_spikes{0, 6, 8};
    ASSERT_EQ(result.potential_, expected_potential);
    ASSERT_EQ(result.spikes_, expected_spikes);
}
