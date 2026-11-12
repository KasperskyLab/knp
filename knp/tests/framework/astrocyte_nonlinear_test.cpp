/**
 * @file astrocyte_nonlinear_test.cpp
 * @brief Tests for nonlinear astrocyte dynamics and serialization.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <knp/astrocyte-traits/all_traits.h>
#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/framework/sonata/network_io.h>

#include <tests_common.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <vector>

#include <highfive/highfive.hpp>

namespace
{
using Model = knp::astrocyte_traits::AstrocyteNonlinear;
using Parameters = knp::astrocyte_traits::astrocyte_parameters<Model>;
using Population = knp::core::Population<Model>;
using Simple = knp::astrocyte_traits::AstrocyteSimple;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;

class TestingBackendST : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

std::vector<Parameters> run_astrocyte(
    const Parameters &parameters, size_t steps, const std::vector<size_t> &spikes = {})
{
    const knp::core::UID population_uid, input_uid;
    Population population{population_uid, [&parameters](size_t) { return parameters; }, 1};
    TestingBackendST backend;
    backend.subscribe<knp::core::messaging::SynapticImpactMessage>(population_uid, {input_uid});
    auto endpoint = backend.get_message_bus().create_endpoint();
    backend.load_populations({population});
    backend._init();
    const auto &state = std::get<Population>(*backend.begin_populations())[0];
    std::vector<Parameters> result;
    result.reserve(steps);
    for (size_t step = 0; step < steps; ++step)
    {
        if (step < spikes.size() && spikes[step] != 0)
        {
            const knp::core::messaging::SynapticImpact impact{
                0, 1.F, knp::synapse_traits::OutputType::EXCITATORY, 0, 0};
            endpoint.send_message(knp::core::messaging::SynapticImpactMessage{
                {input_uid, step},
                knp::core::UID{false},
                population_uid,
                true,
                std::vector<knp::core::messaging::SynapticImpact>(spikes[step], impact)});
        }
        backend._step();
        result.push_back(state);
    }
    return result;
}
}  // namespace

TEST(AstrocyteNonlinear, AmplitudeResponse)
{
    Parameters parameters;
    float previous_amplitude = 0.F;
    for (size_t count : {1, 5, 10, 20})
    {
        const auto states = run_astrocyte(parameters, parameters.window_steps_, {count});
        ASSERT_FALSE(states.empty());
        const auto &state = states.back();
        EXPECT_TRUE(state.pulse_active_);
        EXPECT_GT(state.amplitude_, previous_amplitude);
        EXPECT_LE(state.amplitude_, parameters.amplitude_max_);
        EXPECT_GT(state.calcium_, 0.F);
        EXPECT_LE(state.calcium_, state.amplitude_);
        previous_amplitude = state.amplitude_;
    }
}

TEST(AstrocyteNonlinear, WindowPulseAndIgnoredSpikes)
{
    Parameters parameters;
    const size_t width = parameters.window_steps_;
    const size_t pulse_steps = static_cast<size_t>(parameters.pulse_duration_ / parameters.dt_) + 1;
    const size_t restart = width + pulse_steps - 1;
    std::vector<size_t> spikes(restart + width, 0);
    spikes[0] = 8;
    std::fill(spikes.begin() + width, spikes.begin() + restart, 100);
    spikes[restart] = 3;
    const auto states = run_astrocyte(parameters, spikes.size(), spikes);
    EXPECT_FALSE(states[width - 2].pulse_active_);
    EXPECT_TRUE(states[width - 1].pulse_active_);
    EXPECT_EQ(states[width - 1].activation_count_, 1);
    EXPECT_TRUE(states[restart - 1].pulse_active_);
    EXPECT_EQ(states[restart - 1].total_spikes_, 8);
    EXPECT_EQ(states[restart - 1].spike_counter_, 0);
    EXPECT_FALSE(states[restart].pulse_active_);
    EXPECT_FLOAT_EQ(states[restart].calcium_, 0.F);
    EXPECT_EQ(states[restart].window_step_, 1);
    EXPECT_EQ(states[restart].spike_counter_, 3);
    EXPECT_TRUE(states.back().pulse_active_);
    EXPECT_EQ(states.back().activation_count_, 2);
    EXPECT_EQ(states.back().total_spikes_, 11);
}

TEST(AstrocyteNonlinear, EmptyWindowsDoNotActivate)
{
    Parameters parameters;
    const auto states = run_astrocyte(parameters, 2 * parameters.window_steps_);
    for (const auto &state : states)
    {
        EXPECT_FALSE(state.pulse_active_);
        EXPECT_FLOAT_EQ(state.calcium_, 0.F);
        EXPECT_EQ(state.activation_count_, 0);
    }
}

TEST(AstrocyteNonlinear, SlidingWindowAndFiveStepChecks)
{
    Parameters parameters;
    const size_t width = parameters.window_steps_;
    const size_t stride = parameters.window_stride_;
    std::vector<size_t> spikes(width + stride, 0);
    spikes[width] = 8;
    const auto states = run_astrocyte(parameters, spikes.size(), spikes);
    for (size_t step = 0; step < width + stride - 1; ++step) EXPECT_FALSE(states[step].pulse_active_);
    EXPECT_TRUE(states.back().pulse_active_);
    EXPECT_EQ(states.back().activation_count_, 1);
    EXPECT_EQ(states.back().total_spikes_, 8);
}

TEST(AstrocyteNonlinear, SpikeCapAndTimeScale)
{
    Parameters parameters;
    const size_t width = parameters.window_steps_;
    const size_t duration = static_cast<size_t>(parameters.pulse_duration_ / parameters.dt_);
    const auto limited = run_astrocyte(parameters, width + duration, {parameters.spike_count_max_});
    const auto excessive = run_astrocyte(parameters, width + duration, {10 * parameters.spike_count_max_});
    for (size_t step = width - 1; step < limited.size(); ++step)
    {
        EXPECT_FLOAT_EQ(excessive[step].amplitude_, limited[step].amplitude_);
        EXPECT_FLOAT_EQ(excessive[step].calcium_, limited[step].calcium_);
    }
    EXPECT_EQ(limited.back().total_spikes_, parameters.spike_count_max_);
    EXPECT_EQ(excessive.back().total_spikes_, 10 * parameters.spike_count_max_);
    const auto peak = std::max_element(
        limited.begin(), limited.end(),
        [](const auto &left, const auto &right) { return left.calcium_ < right.calcium_; });
    EXPECT_GT(peak->calcium_, limited[width - 1].calcium_);
    EXPECT_GT(peak->calcium_, limited.back().calcium_);
    EXPECT_LE(peak->calcium_, parameters.amplitude_max_);

    Parameters stretched = parameters;
    stretched.pulse_time_scale_ *= 2.F;
    stretched.pulse_duration_ *= 2.F;
    const auto stretched_states = run_astrocyte(stretched, width + 2 * duration, {parameters.spike_count_max_});
    for (size_t time = 0; time <= duration; ++time)
        EXPECT_FLOAT_EQ(stretched_states[width - 1 + 2 * time].calcium_, limited[width - 1 + time].calcium_);
}

TEST(AstrocyteNonlinear, InvalidTimeParameters)
{
    Parameters parameters;
    parameters.window_steps_ = 0;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
    parameters.window_steps_ = 50;
    parameters.window_stride_ = 0;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
    parameters.window_stride_ = 5;
    parameters.pulse_time_scale_ = 0.F;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
    parameters.pulse_time_scale_ = 100.F;
    parameters.dt_ = 0.F;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
    parameters.dt_ = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
    parameters.dt_ = 1.F;
    parameters.pulse_duration_ = -1.F;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
}

TEST(AstrocyteNonlinear, ModulationThresholdAndPulseEnd)
{
    Parameters parameters;
    parameters.pulse_active_ = true;
    parameters.amplitude_ = parameters.amplitude_max_;
    parameters.pulse_time_ = 2.F * parameters.pulse_time_scale_;
    parameters.active_weight_multiplier_ = 10.F;
    Population population{[parameters](size_t) { return parameters; }, 1};
    const knp::core::UID neuron_uid;
    auto generator = [](size_t) -> std::optional<DeltaProjection::Synapse> {
        return DeltaProjection::Synapse{{0.2F, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
    };
    DeltaProjection controlled{neuron_uid, neuron_uid, generator, 1};
    population.get_tags()[knp::astrocyte_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{controlled.get_uid()};
    DeltaProjection untouched{neuron_uid, neuron_uid, generator, 1};
    TestingBackendST backend;
    backend.load_populations({population});
    backend.load_projections({controlled, untouched});
    backend._init();
    auto &state = std::get<Population>(*backend.begin_populations())[0];
    auto weight = [&backend](size_t index)
    {
        const auto &projection = std::get<DeltaProjection>((backend.begin_projections() + index)->arg_);
        return std::get<knp::core::synapse_data>(projection[0]).weight_;
    };
    backend._step();
    ASSERT_GE(state.calcium_, parameters.modulation_threshold_);
    EXPECT_FLOAT_EQ(weight(0), 2.F);
    EXPECT_FLOAT_EQ(weight(1), 0.2F);
    state.modulation_threshold_ = state.calcium_;
    state.pulse_time_ = parameters.pulse_time_;
    backend._step();
    EXPECT_FLOAT_EQ(state.calcium_, state.modulation_threshold_);
    EXPECT_FLOAT_EQ(weight(0), 2.F);
    state.modulation_threshold_ += 0.01F;
    state.pulse_time_ = parameters.pulse_time_;
    backend._step();
    EXPECT_FLOAT_EQ(weight(0), 0.2F);
    state.pulse_time_ = parameters.pulse_duration_ + parameters.dt_;
    backend._step();
    EXPECT_FALSE(state.pulse_active_);
    EXPECT_FLOAT_EQ(state.calcium_, 0.F);
    EXPECT_FLOAT_EQ(weight(0), 0.2F);
    EXPECT_FLOAT_EQ(weight(1), 0.2F);
}

TEST(AstrocyteNonlinear, SonataMixedFamiliesAndContinuation)
{
    Parameters parameters;
    parameters.window_steps_ = 4;
    parameters.window_stride_ = 2;
    parameters.window_position_ = 2;
    parameters.window_check_step_ = 1;
    parameters.spike_count_max_ = 17;
    parameters.pulse_time_scale_ = 50.F;
    parameters.spike_history_ = {1, 2, 0, 0};
    parameters.dt_ = 0.5F;
    parameters.pulse_duration_ = 6.F;
    parameters.amplitude_max_ = 0.7F;
    parameters.amplitude_slope_ = 0.4F;
    parameters.spike_midpoint_ = 6.F;
    parameters.rise_slope_ = 4.F;
    parameters.fall_slope_ = 1.5F;
    parameters.rise_time_ = 2.F;
    parameters.fall_time_scale_ = 7.F;
    parameters.modulation_threshold_ = 0.03F;
    parameters.active_weight_multiplier_ = 4.F;
    parameters.min_weight_ = 0.002F;
    parameters.calcium_ = 0.01F;
    parameters.window_step_ = 2;
    parameters.spike_counter_ = 3;
    parameters.total_spikes_ = 15;
    parameters.activation_count_ = 3;
    parameters.monitored_spikes_this_step_ = 2;
    parameters.pulse_active_ = false;
    parameters.pulse_time_ = 0.F;
    parameters.amplitude_ = 0.2F;
    Population nonlinear{[parameters](size_t) { return parameters; }, 1};
    const auto nonlinear_uid = nonlinear.get_uid();
    knp::core::Population<Simple> simple{
        [](size_t) { return knp::astrocyte_traits::astrocyte_parameters<Simple>{}; }, 1};
    const auto simple_uid = simple.get_uid();
    DeltaProjection projection{
        simple_uid, nonlinear_uid,
        [](size_t) -> std::optional<DeltaProjection::Synapse> {
            return DeltaProjection::Synapse{{1.F, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
        },
        1};
    const auto controlled_uid = projection.get_uid();
    nonlinear.get_tags()[knp::astrocyte_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{controlled_uid};
    knp::framework::Network network;
    network.add_population(std::move(simple));
    network.add_population(std::move(nonlinear));
    network.add_projection(std::move(projection));
    const auto path = std::filesystem::current_path() / ("nonlinear-roundtrip-" + std::string(knp::core::UID{}));
    std::filesystem::create_directory(path);
    knp::framework::sonata::save_network(network, path);
    {
        HighFive::File file((path / "network/populations.h5").string(), HighFive::File::ReadOnly);
        const auto nodes = file.getGroup("nodes");
        EXPECT_EQ(
            nodes.getGroup(std::string(simple_uid)).getDataSet("node_type_id").read<std::vector<size_t>>()[0], 105);
        EXPECT_EQ(
            nodes.getGroup(std::string(nonlinear_uid)).getDataSet("node_type_id").read<std::vector<size_t>>()[0], 106);
    }
    auto restored = knp::framework::sonata::load_network(path);
    ASSERT_EQ(restored.populations_count(), 2);
    ASSERT_EQ(restored.projections_count(), 1);
    const auto &restored_nonlinear = restored.get_population<Model>(nonlinear_uid);
    ASSERT_TRUE(restored_nonlinear.get_tags().exists(knp::astrocyte_traits::astrocyte_modulated_projections_tag));
    EXPECT_EQ(
        restored_nonlinear.get_tags().get_tag<std::vector<knp::core::UID>>(
            knp::astrocyte_traits::astrocyte_modulated_projections_tag),
        std::vector<knp::core::UID>{controlled_uid});
    auto actual = restored_nonlinear[0];
    EXPECT_FLOAT_EQ(actual.calcium_, parameters.calcium_);
    EXPECT_EQ(actual.window_step_, parameters.window_step_);
    EXPECT_EQ(actual.spike_counter_, parameters.spike_counter_);
    EXPECT_EQ(actual.total_spikes_, parameters.total_spikes_);
    EXPECT_EQ(actual.activation_count_, parameters.activation_count_);
    EXPECT_EQ(actual.monitored_spikes_this_step_, parameters.monitored_spikes_this_step_);
    EXPECT_EQ(actual.pulse_active_, parameters.pulse_active_);
    EXPECT_FLOAT_EQ(actual.pulse_time_, parameters.pulse_time_);
    EXPECT_FLOAT_EQ(actual.amplitude_, parameters.amplitude_);
    EXPECT_EQ(actual.window_steps_, parameters.window_steps_);
    EXPECT_EQ(actual.window_stride_, parameters.window_stride_);
    EXPECT_EQ(actual.window_position_, parameters.window_position_);
    EXPECT_EQ(actual.window_check_step_, parameters.window_check_step_);
    EXPECT_EQ(actual.spike_count_max_, parameters.spike_count_max_);
    EXPECT_EQ(actual.pulse_time_scale_, parameters.pulse_time_scale_);
    EXPECT_EQ(actual.spike_history_, parameters.spike_history_);
    EXPECT_FLOAT_EQ(actual.dt_, parameters.dt_);
    EXPECT_FLOAT_EQ(actual.pulse_duration_, parameters.pulse_duration_);
    EXPECT_FLOAT_EQ(actual.amplitude_max_, parameters.amplitude_max_);
    EXPECT_FLOAT_EQ(actual.amplitude_slope_, parameters.amplitude_slope_);
    EXPECT_FLOAT_EQ(actual.spike_midpoint_, parameters.spike_midpoint_);
    EXPECT_FLOAT_EQ(actual.rise_slope_, parameters.rise_slope_);
    EXPECT_FLOAT_EQ(actual.fall_slope_, parameters.fall_slope_);
    EXPECT_FLOAT_EQ(actual.rise_time_, parameters.rise_time_);
    EXPECT_FLOAT_EQ(actual.fall_time_scale_, parameters.fall_time_scale_);
    EXPECT_FLOAT_EQ(actual.modulation_threshold_, parameters.modulation_threshold_);
    EXPECT_FLOAT_EQ(actual.active_weight_multiplier_, parameters.active_weight_multiplier_);
    EXPECT_FLOAT_EQ(actual.min_weight_, parameters.min_weight_);
    const std::vector<size_t> spikes(20, 3);
    const auto expected_states = run_astrocyte(parameters, spikes.size(), spikes);
    const auto actual_states = run_astrocyte(actual, spikes.size(), spikes);
    ASSERT_EQ(actual_states.size(), expected_states.size());
    for (size_t step = 0; step < expected_states.size(); ++step)
    {
        EXPECT_FLOAT_EQ(actual_states[step].calcium_, expected_states[step].calcium_);
        EXPECT_EQ(actual_states[step].activation_count_, expected_states[step].activation_count_);
        EXPECT_EQ(actual_states[step].total_spikes_, expected_states[step].total_spikes_);
    }
    ASSERT_EQ(path.parent_path(), std::filesystem::current_path());
    std::filesystem::remove_all(path);
}
