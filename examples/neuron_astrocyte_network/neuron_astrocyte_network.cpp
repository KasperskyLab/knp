/**
 * @file neuron_astrocyte_network.cpp
 * @brief Simulation of a spiking network with two simple astrocytes and one nonlinear astrocyte.
 * @details Use --output to choose the CSV file, or --help to display command-line options.
 * Without --output, results are saved to stimulation_multi_astro_output.csv in the working directory.
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

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <knp/astrocyte-traits/astrocyte_nonlinear.h>
#include <knp/astrocyte-traits/astrocyte_simple.h>
#include <knp/core/messaging/messaging.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/framework/projection/synapse_generators.h>
#include <knp/framework/tags/name.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using BLIFATPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using SimpleAstrocyte = knp::astrocyte_traits::AstrocyteSimple;
using NonlinearAstrocyte = knp::astrocyte_traits::AstrocyteNonlinear;
using knp::core::messaging::SpikeMessage;

// Post1 neurons 0 and 1 form group A; neurons 2 and 3 form group B.
constexpr size_t pre_size = 3;
constexpr size_t post1_size = 4;
constexpr size_t post2_size = 5;

constexpr float input_weight_min = 1.0F;
constexpr float input_weight_max = 2.0F;
constexpr float pre_loop_weight_min = 0.0005F;
constexpr float pre_loop_weight_max = 0.0015F;
constexpr float synapse_weight_min = 0.05F;
constexpr float synapse_weight_max = 0.09F;

const std::vector<float> astro_weight_multipliers = {10.F, 8.F, 7.F};


inline knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> neuron_generator(
    size_t /*index*/)  // NOLINT(readability/casting)
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>{};
}


inline std::vector<size_t> index_range(size_t n)
{
    std::vector<size_t> result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}


inline std::mt19937 &weight_rng()
{
    static std::mt19937 generator(42);
    return generator;
}


inline auto random_synapse_parameters(float weight_min, float weight_max, uint32_t delay = 1)
{
    return [weight_min, weight_max, delay](size_t, size_t) -> DeltaProjection::SynapseParameters
    {
        std::uniform_real_distribution<float> distribution(weight_min, weight_max);
        const float weight = distribution(weight_rng());
        return {weight, delay, knp::synapse_traits::OutputType::EXCITATORY};
    };
}


inline auto identity_gen_random(float weight_min, float weight_max, uint32_t delay = 1)
{
    return [weight_min, weight_max, delay](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        std::uniform_real_distribution<float> distribution(weight_min, weight_max);
        const float weight = distribution(weight_rng());
        return DeltaProjection::Synapse{{weight, delay, knp::synapse_traits::OutputType::EXCITATORY}, index, index};
    };
}


/**
 * @brief Apply the standard all-to-all connector to selected neuron indices.
 * @param from_indices Source neuron indices.
 * @param to_indices Target neuron indices.
 * @param parameters_generator Generator of synapse parameters.
 * @return Synapse generator using the specified neuron indices.
 * @note The index vectors must outlive the returned generator.
 */
template <typename ParametersGenerator>
inline auto all_to_all_group(
    const std::vector<size_t> &from_indices, const std::vector<size_t> &to_indices,
    ParametersGenerator parameters_generator)
{
    auto generator = knp::framework::projection::synapse_generators::all_to_all<knp::synapse_traits::DeltaSynapse>(
        from_indices.size(), to_indices.size(), parameters_generator);
    return [generator, &from_indices, &to_indices](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        auto synapse = generator(index);
        if (synapse.has_value())
        {
            auto &from = std::get<knp::core::source_neuron_id>(*synapse);
            auto &to = std::get<knp::core::target_neuron_id>(*synapse);
            from = from_indices[from];
            to = to_indices[to];
        }
        return synapse;
    };
}


/**
 * @brief Astrocyte values recorded by the simulation observer.
 */
struct AstrocyteState
{
    float calcium_;
    float multiplier_;
    uint64_t activation_count_;
};


AstrocyteState read_astrocyte_state(knp::core::Backend &backend, const knp::core::UID &astrocyte_uid)
{
    auto ranges = backend.get_network_data();
    for (auto &pop_it = *ranges.population_range.first, &pop_end = *ranges.population_range.second; pop_it != pop_end;
         ++pop_it)
    {
        knp::core::AllPopulationsVariant variant = *pop_it;
        auto state = std::visit(
            [&astrocyte_uid](const auto &population) -> std::optional<AstrocyteState>
            {
                using Model = typename std::decay_t<decltype(population)>::PopulationNeuronType;
                if constexpr (knp::core::is_astrocyte_v<Model>)
                {
                    if (population.get_uid() == astrocyte_uid && population.size() != 0)
                    {
                        const auto &astrocyte = population[0];
                        const bool active = astrocyte.calcium_ >= astrocyte.modulation_threshold_;
                        const float multiplier = active ? astrocyte.active_weight_multiplier_ : 1.F;
                        return AstrocyteState{astrocyte.calcium_, multiplier, astrocyte.activation_count_};
                    }
                }
                return std::nullopt;
            },
            variant);
        if (state.has_value()) return *state;
    }
    throw std::runtime_error("Astrocyte population was not found among the populations loaded into the backend.");
}


std::vector<float> read_projection_weights(knp::core::Backend &backend, const knp::core::UID &projection_uid)
{
    auto ranges = backend.get_network_data();
    for (auto &proj_it = *ranges.projection_range.first, &proj_end = *ranges.projection_range.second;
         proj_it != proj_end; ++proj_it)
    {
        knp::core::AllProjectionsVariant variant = *proj_it;
        if (auto *delta_proj = std::get_if<DeltaProjection>(&variant); delta_proj != nullptr)
        {
            if (delta_proj->get_uid() == projection_uid)
            {
                std::vector<float> weights;
                weights.reserve(delta_proj->size());
                for (const auto &synapse : *delta_proj)
                {
                    weights.push_back(std::get<knp::core::synapse_data>(synapse).weight_);
                }
                return weights;
            }
        }
    }
    throw std::runtime_error("Projection was not found among the projections loaded into the backend.");
}


/**
 * @brief Network with identifiers for the input and observers.
 */
struct NetworkDescription
{
    knp::framework::Network network_;
    knp::core::UID pre_uid_;
    knp::core::UID post1_uid_;
    knp::core::UID post2_uid_;
    knp::core::UID input_uid_;
    std::vector<knp::core::UID> astrocyte_uids_;
    knp::core::UID pre_to_post1_a_uid_;
    knp::core::UID post1_loop_b_uid_;
};


/**
 * @brief Create the neuron and astrocyte populations and their projections.
 * @return Constructed network with identifiers for the input and observers.
 */
NetworkDescription make_network()
{
    const std::vector<size_t> pre_all = index_range(pre_size);
    const std::vector<size_t> post1_group_a = {0, 1};
    const std::vector<size_t> post1_group_b = {2, 3};
    const std::vector<size_t> post2_all = index_range(post2_size);
    const std::vector<size_t> astrocyte_indices{0};

    knp::framework::Network network;
    struct PopulationConfig
    {
        std::string name_;
        size_t size_;
    };
    const std::vector<PopulationConfig> population_configs = {
        {"pre", pre_size}, {"post1", post1_size}, {"post2", post2_size}};
    std::vector<knp::core::UID> population_uids;
    population_uids.reserve(population_configs.size());
    for (const auto &config : population_configs)
    {
        BLIFATPopulation population{neuron_generator, config.size_};
        knp::framework::tags::set_name(population, config.name_);
        population_uids.push_back(population.get_uid());
        network.add_population(std::move(population));
    }
    const auto pre_uid = population_uids[0];
    const auto post1_uid = population_uids[1];
    const auto post2_uid = population_uids[2];

    DeltaProjection input_projection{
        knp::core::UID{false}, pre_uid, identity_gen_random(input_weight_min, input_weight_max), pre_size};
    const auto input_uid = input_projection.get_uid();
    knp::framework::tags::set_name(input_projection, "input");
    network.add_projection(std::move(input_projection));

    const knp::core::UID pre_loop_uid;
    const knp::core::UID pre_to_post1_a_uid;
    const knp::core::UID post1_loop_a_uid;
    const knp::core::UID pre_to_post1_b_uid;
    const knp::core::UID post1_loop_b_uid;
    const knp::core::UID pre_to_post2_uid;
    const knp::core::UID post2_loop_uid;

    struct ProjectionConfig
    {
        std::string name_;
        knp::core::UID uid_;
        knp::core::UID from_uid_;
        knp::core::UID to_uid_;
        const std::vector<size_t> &from_indices_;
        const std::vector<size_t> &to_indices_;
        float weight_min_;
        float weight_max_;
    };

    // Projection order determines which synapses receive each random weight.
    const std::vector<ProjectionConfig> projection_configs = {
        {"pre_loop", pre_loop_uid, pre_uid, pre_uid, pre_all, pre_all, pre_loop_weight_min, pre_loop_weight_max},
        {"pre_to_post1_a", pre_to_post1_a_uid, pre_uid, post1_uid, pre_all, post1_group_a, synapse_weight_min,
         synapse_weight_max},
        {"post1_loop_a", post1_loop_a_uid, post1_uid, post1_uid, post1_group_a, post1_group_a, synapse_weight_min,
         synapse_weight_max},
        {"pre_to_post1_b", pre_to_post1_b_uid, pre_uid, post1_uid, pre_all, post1_group_b, synapse_weight_min,
         synapse_weight_max},
        {"post1_loop_b", post1_loop_b_uid, post1_uid, post1_uid, post1_group_b, post1_group_b, synapse_weight_min,
         synapse_weight_max},
        {"pre_to_post2", pre_to_post2_uid, pre_uid, post2_uid, pre_all, post2_all, synapse_weight_min,
         synapse_weight_max},
        {"post2_loop", post2_loop_uid, post2_uid, post2_uid, post2_all, post2_all, synapse_weight_min,
         synapse_weight_max}};

    for (const auto &config : projection_configs)
    {
        DeltaProjection projection{
            config.uid_, config.from_uid_, config.to_uid_,
            all_to_all_group(
                config.from_indices_, config.to_indices_,
                random_synapse_parameters(config.weight_min_, config.weight_max_)),
            config.from_indices_.size() * config.to_indices_.size()};
        knp::framework::tags::set_name(projection, config.name_);
        network.add_projection(std::move(projection));
    }

    enum class AstrocyteModel
    {
        simple,
        nonlinear
    };

    struct AstrocyteConfig
    {
        AstrocyteModel model_;
        std::string name_;
        std::string pre_projection_name_;
        std::string post_projection_name_;
        float weight_multiplier_;
        knp::core::UID monitored_population_uid_;
        std::vector<size_t> monitored_indices_;
        std::vector<knp::core::UID> modulated_projection_uids_;
    };

    const std::vector<AstrocyteConfig> astrocyte_configs = {
        {AstrocyteModel::simple,
         "astro1",
         "pre_to_astro1",
         "post1_a_to_astro1",
         astro_weight_multipliers[0],
         post1_uid,
         post1_group_a,
         {pre_to_post1_a_uid, post1_loop_a_uid}},
        {AstrocyteModel::simple,
         "astro2",
         "pre_to_astro2",
         "post1_b_to_astro2",
         astro_weight_multipliers[1],
         post1_uid,
         post1_group_b,
         {pre_to_post1_b_uid}},
        {AstrocyteModel::nonlinear,
         "astro3",
         "pre_to_astro3",
         "post2_to_astro3",
         astro_weight_multipliers[2],
         post2_uid,
         post2_all,
         {pre_to_post2_uid, post2_loop_uid}}};

    std::vector<knp::core::UID> astrocyte_uids;
    std::vector<DeltaProjection> astrocyte_projections;
    astrocyte_uids.reserve(astrocyte_configs.size());
    astrocyte_projections.reserve(2 * astrocyte_configs.size());
    const auto astrocyte_synapse_parameters = [](size_t, size_t) -> DeltaProjection::SynapseParameters {
        return {1.0F, 1, knp::synapse_traits::OutputType::EXCITATORY};
    };

    for (const auto &config : astrocyte_configs)
    {
        const knp::core::UID astrocyte_uid;
        const auto add_astrocyte = [&network, &config](auto population)
        {
            population[0].active_weight_multiplier_ = config.weight_multiplier_;
            knp::framework::tags::set_name(population, config.name_);
            population.get_tags()[knp::astrocyte_traits::astrocyte_modulated_projections_tag] =
                config.modulated_projection_uids_;
            network.add_population(std::move(population));
        };
        if (config.model_ == AstrocyteModel::simple)
        {
            add_astrocyte(knp::core::Population<SimpleAstrocyte>{
                astrocyte_uid,
                [](size_t /*index*/) { return knp::astrocyte_traits::astrocyte_parameters<SimpleAstrocyte>{}; }, 1});
        }
        else
        {
            add_astrocyte(knp::core::Population<NonlinearAstrocyte>{
                astrocyte_uid,
                [](size_t /*index*/) { return knp::astrocyte_traits::astrocyte_parameters<NonlinearAstrocyte>{}; }, 1});
        }
        astrocyte_uids.push_back(astrocyte_uid);

        auto &pre_projection = astrocyte_projections.emplace_back(
            pre_uid, astrocyte_uid, all_to_all_group(pre_all, astrocyte_indices, astrocyte_synapse_parameters),
            pre_all.size());
        knp::framework::tags::set_name(pre_projection, config.pre_projection_name_);
        auto &post_projection = astrocyte_projections.emplace_back(
            config.monitored_population_uid_, astrocyte_uid,
            all_to_all_group(config.monitored_indices_, astrocyte_indices, astrocyte_synapse_parameters),
            config.monitored_indices_.size());
        knp::framework::tags::set_name(post_projection, config.post_projection_name_);
    }

    for (auto &projection : astrocyte_projections)
    {
        network.add_projection(std::move(projection));
    }

    return {std::move(network), pre_uid,         post1_uid, post2_uid, input_uid, std::move(astrocyte_uids),
            pre_to_post1_a_uid, post1_loop_b_uid};
}


void run_simulation(
    const std::string &filename, const size_t total_steps = 2000, const size_t stim_end = 50,
    const size_t stim2_start = 500, const size_t stim2_end = 520, const size_t stim_interval = 10)
{
    std::cout << "\n 3 POPULATIONS + 3 ASTROCYTES \n";

    auto network_description = make_network();
    const auto pre_uid = network_description.pre_uid_;
    const auto post1_uid = network_description.post1_uid_;
    const auto post2_uid = network_description.post2_uid_;
    const auto input_uid = network_description.input_uid_;
    const auto astro1_uid = network_description.astrocyte_uids_[0];
    const auto astro2_uid = network_description.astrocyte_uids_[1];
    const auto astro3_uid = network_description.astrocyte_uids_[2];
    const auto pre_to_post1_a_uid = network_description.pre_to_post1_a_uid_;
    const auto post1_loop_b_uid = network_description.post1_loop_b_uid_;

    const knp::core::UID in_ch;
    knp::framework::Model model(std::move(network_description.network_));
    model.add_input_channel(in_ch, input_uid);

    auto input_gen = [stim_end, stim2_start, stim2_end,
                      stim_interval](knp::core::Step step) -> knp::core::messaging::SpikeData
    {
        knp::core::messaging::SpikeData spikes;
        if ((step < stim_end || (step >= stim2_start && step < stim2_end)) && step % stim_interval == 0)
        {
            for (size_t i = 0; i < pre_size; ++i) spikes.push_back(i);
        }
        return spikes;
    };

    knp::framework::BackendLoader backend_loader;

    size_t pre_spike_count_this_step = 0;
    size_t pre_spike_total = 0;
    std::vector<size_t> pre_per_neuron_totals(pre_size, 0);
    size_t post1_group_a_count_this_step = 0;
    size_t post1_group_b_count_this_step = 0;
    size_t post1_group_a_total = 0;
    size_t post1_group_b_total = 0;
    std::vector<size_t> post1_per_neuron_totals(post1_size, 0);
    size_t post2_spike_count_this_step = 0;
    size_t post2_spike_total = 0;
    std::vector<size_t> post2_per_neuron_totals(post2_size, 0);
    size_t step_counter = 0;

    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load_by_name("knp-cpu-single-threaded-backend"), {{in_ch, input_gen}});
    auto backend = model_executor.get_backend();

    // Astrocyte 1 modulates pre_to_post1_a; post1_loop_b provides unmodulated weights for comparison.
    std::cout << "\nLegend for the individual synapse columns in the CSV:\n";
    std::cout << "  PreToPost1A_S0..S5 (modulated by astrocyte 1, x" << astro_weight_multipliers[0] << "): "
              << "pre0->A0, pre1->A0, pre2->A0, pre0->A1, pre1->A1, pre2->A1\n";
    std::cout << "  Post1LoopB_S0..S3 (NOT modulated, weight fixed permanently): "
              << "post1(2)->post1(2), post1(3)->post1(2), post1(2)->post1(3), post1(3)->post1(3)\n";

    std::ofstream data_file(filename);
    if (!data_file)
    {
        throw std::runtime_error("Could not open output CSV file: " + filename);
    }
    data_file << "Step,PreSpikeCount,Post1GroupASpikeCount,Post1GroupBSpikeCount,Post2SpikeCount,"
              << "Astro1_Ca,Astro1_Multiplier,Astro1_Activations,"
              << "Astro2_Ca,Astro2_Multiplier,Astro2_Activations,"
              << "Astro3_Ca,Astro3_Multiplier,Astro3_Activations,"
              << "PreToPost1A_S0,PreToPost1A_S1,PreToPost1A_S2,PreToPost1A_S3,PreToPost1A_S4,PreToPost1A_S5,"
              << "Post1LoopB_S0,Post1LoopB_S1,Post1LoopB_S2,Post1LoopB_S3\n";

    model_executor.add_observer<SpikeMessage>(
        [&pre_spike_count_this_step, &pre_spike_total,
         &pre_per_neuron_totals](const std::vector<SpikeMessage> &messages)
        {
            size_t count = 0;
            for (const auto &msg : messages)
            {
                for (auto idx : msg.neuron_indexes_)
                {
                    ++count;
                    if (idx < pre_per_neuron_totals.size()) ++pre_per_neuron_totals[idx];
                }
            }
            pre_spike_count_this_step = count;
            pre_spike_total += count;
        },
        {pre_uid});

    model_executor.add_observer<SpikeMessage>(
        [&post1_group_a_count_this_step, &post1_group_b_count_this_step, &post1_group_a_total, &post1_group_b_total,
         &post1_per_neuron_totals](const std::vector<SpikeMessage> &messages)
        {
            size_t a_count = 0;
            size_t b_count = 0;
            for (const auto &msg : messages)
            {
                for (auto idx : msg.neuron_indexes_)
                {
                    if (idx < post1_per_neuron_totals.size()) ++post1_per_neuron_totals[idx];
                    if (idx == 0 || idx == 1) ++a_count;
                    if (idx == 2 || idx == 3) ++b_count;
                }
            }
            post1_group_a_count_this_step = a_count;
            post1_group_b_count_this_step = b_count;
            post1_group_a_total += a_count;
            post1_group_b_total += b_count;
        },
        {post1_uid});

    model_executor.add_observer<SpikeMessage>(
        [&post2_spike_count_this_step, &post2_spike_total,
         &post2_per_neuron_totals](const std::vector<SpikeMessage> &messages)
        {
            size_t count = 0;
            for (const auto &msg : messages)
            {
                for (auto idx : msg.neuron_indexes_)
                {
                    ++count;
                    if (idx < post2_per_neuron_totals.size()) ++post2_per_neuron_totals[idx];
                }
            }
            post2_spike_count_this_step = count;
            post2_spike_total += count;
        },
        {post2_uid});

    // Record results after the spike observers update the counters for this step.
    model_executor.add_observer<SpikeMessage>(
        [&data_file, backend, astro1_uid, astro2_uid, astro3_uid, pre_to_post1_a_uid, post1_loop_b_uid, &step_counter,
         &pre_spike_count_this_step, &post1_group_a_count_this_step, &post1_group_b_count_this_step,
         &post2_spike_count_this_step](const std::vector<SpikeMessage> &)
        {
            const size_t step = step_counter++;
            const auto astro1 = read_astrocyte_state(*backend, astro1_uid);
            const auto astro2 = read_astrocyte_state(*backend, astro2_uid);
            const auto astro3 = read_astrocyte_state(*backend, astro3_uid);
            const auto pre_to_post1_a_weights = read_projection_weights(*backend, pre_to_post1_a_uid);
            const auto post1_loop_b_weights = read_projection_weights(*backend, post1_loop_b_uid);

            data_file << step << ',' << pre_spike_count_this_step << ',' << post1_group_a_count_this_step << ','
                      << post1_group_b_count_this_step << ',' << post2_spike_count_this_step << ',' << astro1.calcium_
                      << ',' << astro1.multiplier_ << ',' << astro1.activation_count_ << ',' << astro2.calcium_ << ','
                      << astro2.multiplier_ << ',' << astro2.activation_count_ << ',' << astro3.calcium_ << ','
                      << astro3.multiplier_ << ',' << astro3.activation_count_;
            for (float weight : pre_to_post1_a_weights) data_file << ',' << weight;
            for (float weight : post1_loop_b_weights) data_file << ',' << weight;
            data_file << '\n';

            if (step % 100 == 0)
            {
                std::cout << "Step " << step << ": A1(Ca=" << astro1.calcium_ << ",x" << astro1.multiplier_
                          << ") A2(Ca=" << astro2.calcium_ << ",x" << astro2.multiplier_
                          << ") A3(Ca=" << astro3.calcium_ << ",x" << astro3.multiplier_ << ")\n";
                std::cout << "         PreToPost1A weights: [";
                for (size_t i = 0; i < pre_to_post1_a_weights.size(); ++i)
                    std::cout << (i ? ", " : "") << pre_to_post1_a_weights[i];
                std::cout << "]  Post1LoopB weights: [";
                for (size_t i = 0; i < post1_loop_b_weights.size(); ++i)
                    std::cout << (i ? ", " : "") << post1_loop_b_weights[i];
                std::cout << "]\n";
            }
        },
        {astro3_uid});

    model_executor.start([total_steps](knp::core::Step step) { return step < total_steps; });

    std::cout << "  Pre total spikes (all " << pre_size << " neurons): " << pre_spike_total << "\n";
    for (size_t i = 0; i < pre_size; ++i)
        std::cout << "    pre neuron " << i << ": " << pre_per_neuron_totals[i] << " spikes\n";

    std::cout << "  Post1 group A total spikes (neurons 0,1): " << post1_group_a_total << "\n";
    std::cout << "  Post1 group B total spikes (neurons 2,3): " << post1_group_b_total << "\n";
    for (size_t i = 0; i < post1_size; ++i)
        std::cout << "    post1 neuron " << i << ": " << post1_per_neuron_totals[i] << " spikes\n";

    std::cout << "  Post2 total spikes (all " << post2_size << " neurons): " << post2_spike_total << "\n";
    for (size_t i = 0; i < post2_size; ++i)
        std::cout << "    post2 neuron " << i << ": " << post2_per_neuron_totals[i] << " spikes\n";

    std::cout << "  Data saved to: " << filename << "\n";
}


int main(int argc, char **argv)
{
    try
    {
#ifdef _WIN32
        system("chcp 65001 > nul");
#endif

        po::options_description options("Options");
        options.add_options()("help,h", "Produce help message.")(
            "output,o", po::value<std::string>()->default_value("stimulation_multi_astro_output.csv"),
            "Output CSV file.");
        po::variables_map options_map;
        po::store(po::parse_command_line(argc, argv, options), options_map);
        po::notify(options_map);
        if (options_map.count("help"))
        {
            std::cout << options << std::endl;
            return EXIT_SUCCESS;
        }

        std::cout << "==========================\n";
        std::cout << "   3 POPULATIONS + 3 INDEPENDENT ASTROCYTES\n";
        std::cout
            << "   - Astro1 (Simple): watches pre + post1[0,1]; controls pre->post1[0,1] and its recurrent loop\n";
        std::cout << "   - Astro2 (Simple): watches pre + post1[2,3]; controls ONLY pre->post1[2,3] (input)\n";
        std::cout
            << "   - Astro3 (Nonlinear): watches pre + post2[all]; controls pre->post2 and post2's recurrent loop\n";
        std::cout << "==========================\n\n";

        run_simulation(options_map["output"].as<std::string>());

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
