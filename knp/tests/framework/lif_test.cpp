/**
 * @file lif_test.cpp
 * @brief LIF neuron test.
 * @kaspersky_support David P.
 * @date 05.05.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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
#include <knp/neuron-traits/lif.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>

#include <algorithm>


using Synapse = knp::synapse_traits::DeltaSynapse;


namespace
{

class TestingBackendST : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    TestingBackendST() = default;
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

using Population = knp::core::Population<knp::neuron_traits::LIFNeuron>;
using Projection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;


struct NeuronLog
{
    std::vector<float> potential_;
    std::vector<size_t> spikes_;
};


NeuronLog run_lif_neuron(
    const knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron> &neuron, size_t steps,
    const std::vector<float> &impacts = {})
{
    const knp::core::UID pop_uid, in_uid, out_uid;
    knp::core::Population<knp::neuron_traits::LIFNeuron> population{pop_uid, [&neuron](size_t) { return neuron; }, 1};
    TestingBackendST backend;
    backend.subscribe<knp::core::messaging::SynapticImpactMessage>(pop_uid, {in_uid});
    auto endpoint = backend.get_message_bus().create_endpoint();
    endpoint.subscribe<knp::core::messaging::SpikeMessage>(out_uid, {pop_uid});

    backend.load_populations({population});
    backend._init();
    auto &pop = *backend.begin_populations();
    NeuronLog result;
    const auto &neuron_ref = std::get<knp::core::Population<knp::neuron_traits::LIFNeuron>>(pop)[0];
    for (size_t step = 0; step < steps; ++step)
    {
        const knp::core::messaging::MessageHeader header{in_uid, step};
        if (step < impacts.size())
        {
            knp::core::messaging::SynapticImpact impact{
                0, impacts[step], knp::synapse_traits::OutputType::EXCITATORY, 0, 0};
            const knp::core::messaging::SynapticImpactMessage msg{
                header, knp::core::UID{false}, pop_uid, true, {impact}};
            endpoint.send_message(msg);
        }
        result.potential_.push_back(neuron_ref.potential_);
        backend._step();
        endpoint.receive_all_messages();
        auto out_msgs = endpoint.unload_messages<knp::core::messaging::SpikeMessage>(out_uid);
        if (!out_msgs.empty() && !out_msgs[0].neuron_indexes_.empty())
        {
            if (std::find(out_msgs[0].neuron_indexes_.begin(), out_msgs[0].neuron_indexes_.end(), 0) !=
                out_msgs[0].neuron_indexes_.end())
                result.spikes_.push_back(step);
        }
    }
    return result;
}

}  //namespace


TEST(LIFNeuron, NeuronPotentialLeakRev)
{
    constexpr int starting_potential = 100;
    constexpr float leak_coefficient = 0.25F;
    constexpr size_t steps_amount = 10;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron>{};
    base_neuron.activation_threshold_ = starting_potential + 1;  // We don't want activations in this test.
    base_neuron.leak_coefficient_ = leak_coefficient;
    base_neuron.potential_ = starting_potential;

    auto results = run_lif_neuron(base_neuron, steps_amount);

    std::vector<float> expected_results;
    expected_results.reserve(steps_amount);
    expected_results.push_back(starting_potential);
    for (size_t power = 0; power < steps_amount - 1; ++power)
    {
        expected_results.push_back(expected_results[power] * leak_coefficient);
    }

    ASSERT_EQ(results.potential_, expected_results);
    ASSERT_TRUE(results.spikes_.empty());
}


TEST(LIFNeuron, Threshold)
{
    constexpr float leak_coefficient = 0.5F;
    constexpr float activation_threshold = 1.F;
    constexpr float potential = 3.F;
    constexpr size_t steps_amount = 3;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron>{};
    base_neuron.leak_coefficient_ = leak_coefficient;
    base_neuron.activation_threshold_ = activation_threshold;
    base_neuron.potential_ = potential;
    auto result = run_lif_neuron(base_neuron, steps_amount);
    const std::vector<float> expected_potential{potential, 0, 0};
    const std::vector<size_t> expected_spikes{0};
    ASSERT_EQ(result.potential_, expected_potential);
    ASSERT_EQ(result.spikes_, expected_spikes);
}


TEST(LIFNeuron, ImpactsSpikes)
{
    constexpr float leak_coefficient = 0.5F;
    constexpr float threshold = 6.F;
    constexpr size_t steps_amount = 6;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron>{};
    base_neuron.leak_coefficient_ = leak_coefficient;
    base_neuron.activation_threshold_ = threshold;
    const std::vector<float> impacts{5.F, 5.F, 2.F, 3.F, 8.F};
    auto result = run_lif_neuron(base_neuron, steps_amount, impacts);
    const std::vector<float> expected_potential{0.F, 5.F, 0.F, 2.F, 4.F, 0.F};
    const std::vector<size_t> expected_spikes{1, 4};
    ASSERT_EQ(result.potential_, expected_potential);
    ASSERT_EQ(result.spikes_, expected_spikes);
}


TEST(LIFNeuron, RefractPeriod)
{
    constexpr float leak_coefficient = 0.5F;
    constexpr float threshold = 6.F;
    constexpr size_t steps_amount = 7;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron>{};
    base_neuron.leak_coefficient_ = leak_coefficient;
    base_neuron.activation_threshold_ = threshold;
    base_neuron.refract_period_ = 2;
    const std::vector<float> impacts{5.F, 5.F, 10.F, 10.F, 5.F, 5.F};
    auto result = run_lif_neuron(base_neuron, steps_amount, impacts);
    const std::vector<float> expected_potential{0.F, 5.F, 0.F, 0.F, 0.F, 5.F, 0.F};
    const std::vector<size_t> expected_spikes{1, 5};
    ASSERT_EQ(result.potential_, expected_potential);
    ASSERT_EQ(result.spikes_, expected_spikes);
}
