/**
 * @file astrocyte_separation_test.cpp
 * @brief Tests for separate neuron and astrocyte families.
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
#include <knp/framework/normalization/neuron_accessors.h>
#include <knp/framework/normalization/neuron_normalizer.h>
#include <knp/framework/sonata/network_io.h>

#include <tests_common.h>

#include <algorithm>
#include <filesystem>
#include <type_traits>
#include <vector>

#include <highfive/highfive.hpp>

namespace
{

using Astrocyte = knp::astrocyte_traits::AstrocyteSimple;
using AstrocyteParameters = knp::astrocyte_traits::astrocyte_parameters<Astrocyte>;
using AstrocytePopulation = knp::core::Population<Astrocyte>;
using Neuron = knp::neuron_traits::BLIFATNeuron;
using NeuronPopulation = knp::core::Population<Neuron>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;

static_assert(knp::core::is_astrocyte_v<Astrocyte>);
static_assert(!knp::core::is_neuron_v<Astrocyte>);
static_assert(std::is_same_v<AstrocytePopulation::NeuronParameters, AstrocyteParameters>);
static_assert(std::is_same_v<NeuronPopulation::NeuronParameters, knp::neuron_traits::neuron_parameters<Neuron>>);

// Bindings require Population member-function pointer types.
static_assert(std::is_same_v<
              decltype(&NeuronPopulation::remove_neurons), void (NeuronPopulation::*)(const std::vector<size_t> &)>);
static_assert(std::is_same_v<decltype(&NeuronPopulation::remove_neuron), void (NeuronPopulation::*)(const size_t &)>);

template <typename T, typename = void>
struct HasMembranePotential : std::false_type
{
};

template <typename T>
struct HasMembranePotential<T, std::void_t<decltype(std::declval<T>().potential_)>> : std::true_type
{
};

static_assert(HasMembranePotential<NeuronPopulation::NeuronParameters>::value);
static_assert(!HasMembranePotential<AstrocyteParameters>::value);

class TestingBackendST : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

std::vector<AstrocyteParameters> run_astrocyte(
    const AstrocyteParameters &parameters, size_t steps, const std::vector<size_t> &spikes = {})
{
    const knp::core::UID population_uid, input_uid;
    AstrocytePopulation population{population_uid, [&parameters](size_t) { return parameters; }, 1};
    TestingBackendST backend;
    backend.subscribe<knp::core::messaging::SynapticImpactMessage>(population_uid, {input_uid});
    auto endpoint = backend.get_message_bus().create_endpoint();
    backend.load_populations({population});
    backend._init();
    const auto &state = std::get<AstrocytePopulation>(*backend.begin_populations())[0];
    std::vector<AstrocyteParameters> result;
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

template <class Model>
void check_active_modulation_roundtrip(const std::filesystem::path &path)
{
    using Population = knp::core::Population<Model>;
    using Parameters = knp::astrocyte_traits::astrocyte_parameters<Model>;
    Parameters parameters;
    parameters.active_weight_multiplier_ = 2.F;
    if constexpr (std::is_same_v<Model, knp::astrocyte_traits::AstrocyteSimple>)
    {
        parameters.calcium_ = 1.F;
        parameters.calcium_step_ = 0.1F;
    }
    else
    {
        parameters.pulse_active_ = true;
        parameters.amplitude_ = parameters.amplitude_max_;
        parameters.pulse_time_ = 2.F * parameters.pulse_time_scale_;
    }
    Population population{[parameters](size_t) { return parameters; }, 1};
    NeuronPopulation neurons{[](size_t) { return knp::neuron_traits::neuron_parameters<Neuron>{}; }, 1};
    DeltaProjection projection{
        neurons.get_uid(), neurons.get_uid(),
        [](size_t index) -> std::optional<DeltaProjection::Synapse>
        {
            return DeltaProjection::Synapse{
                {0.1F * static_cast<float>(index + 1), 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
        },
        3};
    const auto &tag = knp::astrocyte_traits::astrocyte_modulated_projections_tag;
    const auto &weights_tag = knp::astrocyte_traits::astrocyte_original_weights_tag;
    population.get_tags()[tag] = std::vector<knp::core::UID>{projection.get_uid()};
    TestingBackendST original;
    original.load_populations({population});
    original.load_projections({projection});
    original._init();
    original._step();
    knp::framework::Network snapshot;
    snapshot.add_population(std::get<Population>(*original.begin_populations()));
    snapshot.add_population(std::move(neurons));
    snapshot.add_projection(std::get<DeltaProjection>(original.begin_projections()->arg_));
    std::filesystem::create_directory(path);
    knp::framework::sonata::save_network(snapshot, path);
    const auto restored = knp::framework::sonata::load_network(path);
    const auto &restored_population = restored.get_population<Model>(population.get_uid());
    ASSERT_TRUE(restored_population.get_tags().exists(tag));
    EXPECT_EQ(
        restored_population.get_tags().template get_tag<std::vector<knp::core::UID>>(tag),
        std::vector<knp::core::UID>{projection.get_uid()});
    ASSERT_TRUE(restored_population.get_tags().exists(weights_tag));
    const auto &weights = restored_population.get_tags()
                              .template get_tag<knp::astrocyte_traits::AstrocyteProjectionWeights>(weights_tag)
                              .at(std::string(projection.get_uid()));
    ASSERT_EQ(weights.size(), projection.size());
    for (size_t index = 0; index < weights.size(); ++index)
        EXPECT_FLOAT_EQ(weights[index], std::get<knp::core::synapse_data>(projection[index]).weight_);

    TestingBackendST loaded;
    loaded.load_all_populations(restored.get_populations());
    loaded.load_all_projections(restored.get_projections());
    loaded._init();
    original._step();
    loaded._step();
    const auto &continued = std::get<DeltaProjection>(original.begin_projections()->arg_);
    const auto &reloaded = std::get<DeltaProjection>(loaded.begin_projections()->arg_);
    for (size_t index = 0; index < weights.size(); ++index)
    {
        EXPECT_FLOAT_EQ(
            std::get<knp::core::synapse_data>(reloaded[index]).weight_,
            std::get<knp::core::synapse_data>(continued[index]).weight_);
        EXPECT_FLOAT_EQ(
            std::get<knp::core::synapse_data>(reloaded[index]).weight_,
            weights[index] * parameters.active_weight_multiplier_);
    }
    const auto loaded_population = std::find_if(
        loaded.begin_populations(), loaded.end_populations(),
        [](const auto &value) { return std::holds_alternative<Population>(value); });
    ASSERT_NE(loaded_population, loaded.end_populations());
    auto &state = std::get<Population>(*loaded_population)[0];
    if constexpr (std::is_same_v<Model, knp::astrocyte_traits::AstrocyteSimple>)
        state.calcium_ = 0.F;
    else
        state.pulse_time_ = state.pulse_duration_ + state.dt_;
    loaded._step();
    for (size_t index = 0; index < weights.size(); ++index)
        EXPECT_FLOAT_EQ(std::get<knp::core::synapse_data>(reloaded[index]).weight_, weights[index]);
}

}  // namespace

TEST(AstrocyteFamilies, SeparateRegistriesAndPopulationAPI)
{
    EXPECT_FALSE((boost::mp11::mp_contains<knp::neuron_traits::AllNeurons, Astrocyte>::value));
    EXPECT_TRUE((boost::mp11::mp_contains<knp::astrocyte_traits::AllAstrocytes, Astrocyte>::value));
    EXPECT_EQ(
        boost::mp11::mp_size<knp::core::AllCells>::value,
        boost::mp11::mp_size<knp::neuron_traits::AllNeurons>::value +
            boost::mp11::mp_size<knp::astrocyte_traits::AllAstrocytes>::value);

    AstrocytePopulation population{[](size_t) { return AstrocyteParameters{}; }, 2};
    auto parameters = population.get_neurons_parameters();
    parameters[1].calcium_ = 17.F;
    population.set_neurons_parameters(parameters);
    EXPECT_FLOAT_EQ(population.get_neuron_parameters(1).calcium_, 17.F);
    population.add_neurons([](size_t) { return AstrocyteParameters{}; }, 1);
    population.remove_neuron(0);
    EXPECT_EQ(population.size(), 2);
    EXPECT_FLOAT_EQ(population[0].calcium_, 17.F);
}

TEST(AstrocyteFamilies, BackendSeparatesCellRegistries)
{
    using Backend = knp::backends::single_threaded_cpu::SingleThreadedCPUBackend;
    EXPECT_TRUE((boost::mp11::mp_contains<Backend::SupportedAstrocytes, Astrocyte>::value));
    EXPECT_FALSE((boost::mp11::mp_contains<Backend::SupportedNeurons, Astrocyte>::value));

    Backend backend;
    const auto neurons = backend.get_supported_neurons();
    EXPECT_EQ(std::find(neurons.begin(), neurons.end(), "AstrocyteSimple"), neurons.end());
}

TEST(AstrocyteFamilies, NeuronNormalizationPreservesAstrocyteState)
{
    NeuronPopulation neurons{
        [](size_t index)
        {
            auto parameters = knp::neuron_traits::neuron_parameters<Neuron>{};
            parameters.potential_ = 2.F + 2.F * static_cast<float>(index);
            return parameters;
        },
        2};
    const auto neuron_uid = neurons.get_uid();
    AstrocytePopulation astrocytes{
        [](size_t /*index*/)  // NOLINT(readability/casting)
        {
            auto parameters = AstrocyteParameters{};
            parameters.calcium_ = 23.F;
            return parameters;
        },
        1};
    const auto astrocyte_uid = astrocytes.get_uid();
    knp::framework::Network network;
    network.add_population(std::move(neurons));
    network.add_population(std::move(astrocytes));

    knp::framework::normalization::normalize_neurons<
        knp::framework::normalization::PotentialAccessor, knp::framework::normalization::PotentialAccessor>(
        network, knp::framework::normalization::Rescaler<float>(2.F, 4.F, 0.F, 1.F));

    const auto &normalized = network.get_population<Neuron>(neuron_uid);
    EXPECT_FLOAT_EQ(normalized[0].potential_, 0.F);
    EXPECT_FLOAT_EQ(normalized[1].potential_, 1.F);
    EXPECT_FLOAT_EQ(network.get_population<Astrocyte>(astrocyte_uid)[0].calcium_, 23.F);
}

TEST(AstrocyteDynamics, AccumulationRiseDecayAndReactivation)
{
    AstrocyteParameters parameters;
    parameters.spike_threshold_ = 3;
    parameters.window_steps_ = 2;
    parameters.window_stride_ = 1;
    parameters.calcium_max_ = 2.F;
    parameters.calcium_step_ = 1.F;
    const std::vector<size_t> spikes{2, 1, 100, 100, 100, 3, 0};
    const std::vector<float> expected_calcium{0.F, 1.F, 2.F, 1.F, 0.F, 0.F, 1.F};
    const auto states = run_astrocyte(parameters, spikes.size(), spikes);
    ASSERT_EQ(states.size(), expected_calcium.size());
    for (size_t step = 0; step < states.size(); ++step) EXPECT_FLOAT_EQ(states[step].calcium_, expected_calcium[step]);
    EXPECT_EQ(states[0].spike_counter_, 2);
    EXPECT_EQ(states[1].activation_count_, 1);
    EXPECT_FALSE(states[2].rising_);
    EXPECT_EQ(states[4].total_spikes_, 3);
    EXPECT_EQ(states[5].spike_counter_, 3);
    EXPECT_EQ(states.back().activation_count_, 2);
    EXPECT_EQ(states.back().total_spikes_, 6);
}

TEST(AstrocyteDynamics, SlidingWindowExpiresSpikesAndChecksEveryFiveSteps)
{
    AstrocyteParameters parameters;
    parameters.calcium_max_ = 2.F;
    parameters.calcium_step_ = 1.F;
    const size_t width = parameters.window_steps_;
    const size_t activation = width + parameters.window_stride_ - 1;
    const size_t pulse_end = activation + 3;
    std::vector<size_t> spikes(pulse_end + width + 1, 0);
    spikes[0] = parameters.spike_threshold_ - 1;
    spikes[width] = 1;
    spikes[width + 1] = parameters.spike_threshold_ - 1;
    std::fill(spikes.begin() + activation + 1, spikes.begin() + pulse_end + 1, 100);
    const auto states = run_astrocyte(parameters, spikes.size(), spikes);
    for (size_t step = 0; step < activation; ++step) EXPECT_FLOAT_EQ(states[step].calcium_, 0.F);
    EXPECT_EQ(states[width].spike_counter_, 1);
    EXPECT_EQ(states[width + 1].spike_counter_, parameters.spike_threshold_);
    EXPECT_FLOAT_EQ(states[activation].calcium_, parameters.calcium_step_);
    EXPECT_EQ(states[activation].activation_count_, 1);
    EXPECT_FLOAT_EQ(states[pulse_end].calcium_, 0.F);
    EXPECT_EQ(states[pulse_end].total_spikes_, 2 * parameters.spike_threshold_ - 1);
    EXPECT_EQ(states.back().activation_count_, 1);
    EXPECT_EQ(states.back().spike_counter_, 0);
}

TEST(AstrocyteDynamics, FullWindowBeforeActivationAndInvalidWindowParameters)
{
    AstrocyteParameters parameters;
    const auto states = run_astrocyte(parameters, parameters.window_steps_, {parameters.spike_threshold_});
    for (size_t step = 0; step + 1 < states.size(); ++step) EXPECT_FLOAT_EQ(states[step].calcium_, 0.F);
    EXPECT_FLOAT_EQ(states.back().calcium_, parameters.calcium_step_);
    EXPECT_EQ(states.back().activation_count_, 1);
    parameters.window_steps_ = 0;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
    parameters.window_steps_ = 50;
    parameters.window_stride_ = 0;
    EXPECT_THROW(run_astrocyte(parameters, 1), std::invalid_argument);
}

TEST(AstrocyteDynamics, ModulationIsSelectiveAndDoesNotCompound)
{
    AstrocyteParameters parameters;
    parameters.calcium_ = 0.1F;
    parameters.rising_ = true;
    parameters.calcium_step_ = 0.1F;
    parameters.active_weight_multiplier_ = 10.F;
    AstrocytePopulation population{[parameters](size_t) { return parameters; }, 1};
    const knp::core::UID neuron_uid;
    auto generator = [](size_t) -> std::optional<DeltaProjection::Synapse> {
        return DeltaProjection::Synapse{{0.2F, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
    };
    DeltaProjection controlled{neuron_uid, neuron_uid, generator, 1};
    DeltaProjection untouched{neuron_uid, neuron_uid, generator, 1};
    population.get_tags()[knp::astrocyte_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{controlled.get_uid()};
    TestingBackendST backend;
    backend.load_populations({population});
    backend.load_projections({controlled, untouched});
    backend._init();
    auto &state = std::get<AstrocytePopulation>(*backend.begin_populations())[0];
    auto weight = [&backend](size_t index)
    {
        const auto &projection = std::get<DeltaProjection>((backend.begin_projections() + index)->arg_);
        return std::get<knp::core::synapse_data>(projection[0]).weight_;
    };
    backend._step();
    EXPECT_FLOAT_EQ(state.calcium_, parameters.modulation_threshold_);
    EXPECT_FLOAT_EQ(weight(0), 2.F);
    EXPECT_FLOAT_EQ(weight(1), 0.2F);
    backend._step();
    EXPECT_FLOAT_EQ(weight(0), 2.F);
    state.calcium_ = parameters.modulation_threshold_;
    state.rising_ = false;
    backend._step();
    EXPECT_LT(state.calcium_, parameters.modulation_threshold_);
    EXPECT_FLOAT_EQ(weight(0), 0.2F);
    EXPECT_FLOAT_EQ(weight(1), 0.2F);
}

class AstrocyteSonata : public ::testing::Test
{
protected:
    void SetUp() override
    {
        path_ = std::filesystem::current_path() / ("astrocyte-roundtrip-" + std::string(knp::core::UID{}));
        std::filesystem::create_directory(path_);
    }

    void TearDown() override
    {
        if (HasFailure()) return;
        ASSERT_EQ(path_.parent_path(), std::filesystem::current_path());
        ASSERT_EQ(path_.filename().string().find("astrocyte-roundtrip-"), 0);
        std::filesystem::remove_all(path_);
    }

    std::filesystem::path path_;
};

TEST_F(AstrocyteSonata, RoundTripParametersAndLegacyTypeId)
{
    AstrocyteParameters parameters;
    parameters.calcium_ = 33.F;
    parameters.rising_ = true;
    parameters.spike_counter_ = 9;
    parameters.window_steps_ = 4;
    parameters.window_stride_ = 2;
    parameters.window_step_ = 4;
    parameters.window_position_ = 2;
    parameters.window_check_step_ = 1;
    parameters.spike_history_ = {2, 3, 1, 3};
    parameters.total_spikes_ = 50;
    parameters.activation_count_ = 2;
    parameters.monitored_spikes_this_step_ = 3;
    parameters.calcium_max_ = 123.F;
    parameters.calcium_step_ = 2.F;
    parameters.modulation_threshold_ = 4.F;
    parameters.spike_threshold_ = 7;
    parameters.base_weight_ = 0.12F;
    parameters.active_weight_multiplier_ = 5.F;
    parameters.min_weight_ = 0.002F;
    parameters.max_weight_ = 3.F;
    AstrocytePopulation astrocytes{[parameters](size_t) { return parameters; }, 2};
    const auto uid = astrocytes.get_uid();
    NeuronPopulation neurons{[](size_t) { return knp::neuron_traits::neuron_parameters<Neuron>{}; }, 1};
    const auto neuron_uid = neurons.get_uid();
    DeltaProjection projection{
        neuron_uid, uid,
        [](size_t index) -> std::optional<DeltaProjection::Synapse> {
            return DeltaProjection::Synapse{{0.2F, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, index};
        },
        2};
    const auto projection_uid = projection.get_uid();
    astrocytes.get_tags()[knp::astrocyte_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{projection_uid};
    knp::framework::Network network;
    network.add_population(std::move(neurons));
    network.add_population(std::move(astrocytes));
    network.add_projection(std::move(projection));
    knp::framework::sonata::save_network(network, path_);

    {
        HighFive::File file((path_ / "network/populations.h5").string(), HighFive::File::ReadOnly);
        const auto ids =
            file.getGroup("nodes").getGroup(std::string(uid)).getDataSet("node_type_id").read<std::vector<size_t>>();
        ASSERT_EQ(ids.size(), 2);
        EXPECT_EQ(ids[0], 105);
        EXPECT_EQ(ids[1], 105);
    }
    auto restored = knp::framework::sonata::load_network(path_);
    ASSERT_EQ(restored.populations_count(), 2);
    ASSERT_EQ(restored.projections_count(), 1);
    const auto &restored_projection = restored.get_projection<knp::synapse_traits::DeltaSynapse>(projection_uid);
    EXPECT_EQ(restored_projection.get_presynaptic(), neuron_uid);
    EXPECT_EQ(restored_projection.get_postsynaptic(), uid);
    ASSERT_EQ(restored_projection.size(), 2);
    EXPECT_FLOAT_EQ(std::get<knp::core::synapse_data>(restored_projection[0]).weight_, 0.2F);
    EXPECT_EQ(std::get<knp::core::source_neuron_id>(restored_projection[0]), 0);
    EXPECT_EQ(std::get<knp::core::target_neuron_id>(restored_projection[1]), 1);
    const auto &population = restored.get_population<Astrocyte>(uid);
    ASSERT_EQ(population.size(), 2);
    ASSERT_TRUE(population.get_tags().exists(knp::astrocyte_traits::astrocyte_modulated_projections_tag));
    EXPECT_EQ(
        population.get_tags().get_tag<std::vector<knp::core::UID>>(
            knp::astrocyte_traits::astrocyte_modulated_projections_tag),
        std::vector<knp::core::UID>{projection_uid});
    EXPECT_FALSE(population.get_tags().exists(knp::astrocyte_traits::astrocyte_original_weights_tag));
    for (const auto &actual : population)
    {
        EXPECT_FLOAT_EQ(actual.calcium_, parameters.calcium_);
        EXPECT_FLOAT_EQ(actual.calcium_max_, parameters.calcium_max_);
        EXPECT_FLOAT_EQ(actual.calcium_step_, parameters.calcium_step_);
        EXPECT_FLOAT_EQ(actual.modulation_threshold_, parameters.modulation_threshold_);
        EXPECT_FLOAT_EQ(actual.base_weight_, parameters.base_weight_);
        EXPECT_FLOAT_EQ(actual.active_weight_multiplier_, parameters.active_weight_multiplier_);
        EXPECT_FLOAT_EQ(actual.min_weight_, parameters.min_weight_);
        EXPECT_FLOAT_EQ(actual.max_weight_, parameters.max_weight_);
        EXPECT_EQ(actual.rising_, parameters.rising_);
        EXPECT_EQ(actual.spike_counter_, parameters.spike_counter_);
        EXPECT_EQ(actual.window_steps_, parameters.window_steps_);
        EXPECT_EQ(actual.window_stride_, parameters.window_stride_);
        EXPECT_EQ(actual.window_step_, parameters.window_step_);
        EXPECT_EQ(actual.window_position_, parameters.window_position_);
        EXPECT_EQ(actual.window_check_step_, parameters.window_check_step_);
        EXPECT_EQ(actual.spike_history_, parameters.spike_history_);
        EXPECT_EQ(actual.total_spikes_, parameters.total_spikes_);
        EXPECT_EQ(actual.activation_count_, parameters.activation_count_);
        EXPECT_EQ(actual.monitored_spikes_this_step_, parameters.monitored_spikes_this_step_);
        EXPECT_EQ(actual.spike_threshold_, parameters.spike_threshold_);
    }
}

TEST_F(AstrocyteSonata, ActiveModulationRoundTrip)
{
    check_active_modulation_roundtrip<knp::astrocyte_traits::AstrocyteSimple>(path_ / "simple");
    check_active_modulation_roundtrip<knp::astrocyte_traits::AstrocyteNonlinear>(path_ / "nonlinear");
}
