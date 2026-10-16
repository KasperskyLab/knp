#include <knp/core/messaging/messaging.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/astrocyte.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <variant>
#include <vector>

using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using BLIFATPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using AstrocytePopulation = knp::core::Population<knp::neuron_traits::AstrocyteCell>;
using knp::core::messaging::SpikeMessage;

// Three populations: pre_pop, post1_pop (groups A and B), post2_pop.
constexpr size_t PRE_SIZE = 3;
constexpr size_t POST1_SIZE = 4;
constexpr size_t POST2_SIZE = 5;

// Ranges for randomized initial synapse weights.
constexpr float INPUT_WEIGHT_MIN = 1.0F;
constexpr float INPUT_WEIGHT_MAX = 2.0F;
constexpr float PRE_LOOP_WEIGHT_MIN = 0.0005F;
constexpr float PRE_LOOP_WEIGHT_MAX = 0.0015F;
constexpr float SYNAPSE_WEIGHT_MIN = 0.05F;
constexpr float SYNAPSE_WEIGHT_MAX = 0.09F;

// active_weight_multiplier_ for astrocytes 1, 2, 3.
const std::vector<float> ASTRO_WEIGHT_MULTIPLIERS = {10.F, 8.F, 7.F};

inline knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> neuron_generator(size_t)
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>{};
}

inline auto astrocyte_generator(float active_weight_multiplier)
{
    return [active_weight_multiplier](size_t)
               -> knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell>
    {
        auto params = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell>{};
        params.active_weight_multiplier_ = active_weight_multiplier;
        return params;
    };
}

// {0, 1, ..., n-1}.
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

// All-to-all connectivity with an independent random weight for each synapse.
inline auto dense_gen_random(
    std::vector<size_t> from_indices, std::vector<size_t> to_indices, float weight_min, float weight_max,
    uint32_t delay = 1)
{
    return [from_indices, to_indices, weight_min, weight_max, delay](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        const size_t from = from_indices[index / to_indices.size()];
        const size_t to = to_indices[index % to_indices.size()];
        std::uniform_real_distribution<float> distribution(weight_min, weight_max);
        const float weight = distribution(weight_rng());
        return DeltaProjection::Synapse{{weight, delay, knp::synapse_traits::OutputType::EXCITATORY}, from, to};
    };
}

// One-to-one connectivity with an independent random weight for each synapse.
inline auto identity_gen_random(float weight_min, float weight_max, uint32_t delay = 1)
{
    return [weight_min, weight_max, delay](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        std::uniform_real_distribution<float> distribution(weight_min, weight_max);
        const float weight = distribution(weight_rng());
        return DeltaProjection::Synapse{{weight, delay, knp::synapse_traits::OutputType::EXCITATORY}, index, index};
    };
}

// Same connectivity pattern with a fixed weight, used for neuron-to-astrocyte synapses.
inline auto dense_gen(
    std::vector<size_t> from_indices, std::vector<size_t> to_indices, float weight, uint32_t delay = 1)
{
    return [from_indices, to_indices, weight, delay](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        const size_t from = from_indices[index / to_indices.size()];
        const size_t to = to_indices[index % to_indices.size()];
        return DeltaProjection::Synapse{{weight, delay, knp::synapse_traits::OutputType::EXCITATORY}, from, to};
    };
}

namespace
{
knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> read_astrocyte_state(
    knp::core::Backend &backend, const knp::core::UID &astrocyte_uid)
{
    auto ranges = backend.get_network_data();
    auto &pop_it = *ranges.population_range.first;
    auto &pop_end = *ranges.population_range.second;
    for (; pop_it != pop_end; ++pop_it)
    {
        knp::core::AllPopulationsVariant variant = *pop_it;
        if (auto *astro_pop = std::get_if<AstrocytePopulation>(&variant))
        {
            if (astro_pop->get_uid() == astrocyte_uid) return (*astro_pop)[0];
        }
    }
    throw std::runtime_error("Astrocyte population was not found among the populations loaded into the backend.");
}

std::vector<float> read_projection_weights(knp::core::Backend &backend, const knp::core::UID &projection_uid)
{
    auto ranges = backend.get_network_data();
    auto &proj_it = *ranges.projection_range.first;
    auto &proj_end = *ranges.projection_range.second;
    for (; proj_it != proj_end; ++proj_it)
    {
        knp::core::AllProjectionsVariant variant = *proj_it;
        if (auto *delta_proj = std::get_if<DeltaProjection>(&variant))
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

float astrocyte_multiplier(const knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> &astro)
{
    const bool active = astro.calcium_ > astro.modulation_threshold_;
    return active ? astro.active_weight_multiplier_ : 1.F;
}
}  // namespace

void run_simulation(
    const std::string &filename, size_t TOTAL_STEPS = 2000, size_t STIM_END = 50, size_t STIM2_START = 500,
    size_t STIM2_END = 520, size_t STIM_INTERVAL = 10)
{
    std::cout << "\n 3 POPULATIONS + 3 ASTROCYTES \n";

    const std::vector<size_t> pre_all = index_range(PRE_SIZE);
    const std::vector<size_t> post1_group_a = {0, 1};
    const std::vector<size_t> post1_group_b = {2, 3};
    const std::vector<size_t> post2_all = index_range(POST2_SIZE);

    // Populations.
    BLIFATPopulation pre_pop{neuron_generator, PRE_SIZE};
    knp::core::UID pre_uid = pre_pop.get_uid();

    BLIFATPopulation post1_pop{neuron_generator, POST1_SIZE};
    knp::core::UID post1_uid = post1_pop.get_uid();

    BLIFATPopulation post2_pop{neuron_generator, POST2_SIZE};
    knp::core::UID post2_uid = post2_pop.get_uid();

    DeltaProjection input_proj{
        knp::core::UID{false}, pre_uid, identity_gen_random(INPUT_WEIGHT_MIN, INPUT_WEIGHT_MAX), PRE_SIZE};
    knp::core::UID input_uid = input_proj.get_uid();

    DeltaProjection pre_loop{
        pre_uid, pre_uid, dense_gen_random(pre_all, pre_all, PRE_LOOP_WEIGHT_MIN, PRE_LOOP_WEIGHT_MAX),
        PRE_SIZE * PRE_SIZE};

    // Astrocyte 1's territory: pre_pop -> group A, plus group A's own recurrent loop.
    DeltaProjection pre_to_post1_a{
        pre_uid, post1_uid, dense_gen_random(pre_all, post1_group_a, SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX),
        PRE_SIZE * post1_group_a.size()};
    knp::core::UID pre_to_post1_a_uid = pre_to_post1_a.get_uid();

    DeltaProjection post1_loop_a{
        post1_uid, post1_uid, dense_gen_random(post1_group_a, post1_group_a, SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX),
        post1_group_a.size() * post1_group_a.size()};
    knp::core::UID post1_loop_a_uid = post1_loop_a.get_uid();

    // Astrocyte 2's territory: pre_pop -> group B only. Group B's recurrent loop is not modulated.
    DeltaProjection pre_to_post1_b{
        pre_uid, post1_uid, dense_gen_random(pre_all, post1_group_b, SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX),
        PRE_SIZE * post1_group_b.size()};
    knp::core::UID pre_to_post1_b_uid = pre_to_post1_b.get_uid();

    DeltaProjection post1_loop_b{
        post1_uid, post1_uid, dense_gen_random(post1_group_b, post1_group_b, SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX),
        post1_group_b.size() * post1_group_b.size()};
    knp::core::UID post1_loop_b_uid = post1_loop_b.get_uid();

    // Astrocyte 3's territory: pre_pop -> post2, plus post2's own recurrent loop.
    DeltaProjection pre_to_post2{
        pre_uid, post2_uid, dense_gen_random(pre_all, post2_all, SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX),
        PRE_SIZE * post2_all.size()};
    knp::core::UID pre_to_post2_uid = pre_to_post2.get_uid();

    DeltaProjection post2_loop{
        post2_uid, post2_uid, dense_gen_random(post2_all, post2_all, SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX),
        post2_all.size() * post2_all.size()};
    knp::core::UID post2_loop_uid = post2_loop.get_uid();

    knp::framework::Network network;
    network.add_population(std::move(pre_pop));
    network.add_population(std::move(post1_pop));
    network.add_population(std::move(post2_pop));
    network.add_projection(std::move(input_proj));
    network.add_projection(std::move(pre_loop));
    network.add_projection(std::move(pre_to_post1_a));
    network.add_projection(std::move(post1_loop_a));
    network.add_projection(std::move(pre_to_post1_b));
    network.add_projection(std::move(post1_loop_b));
    network.add_projection(std::move(pre_to_post2));
    network.add_projection(std::move(post2_loop));

    // Three astrocytes, each its own population with its own monitored neurons and controlled projections.
    AstrocytePopulation astro1_pop{astrocyte_generator(ASTRO_WEIGHT_MULTIPLIERS[0]), 1};
    knp::core::UID astro1_uid = astro1_pop.get_uid();
    astro1_pop.get_tags()[knp::neuron_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{pre_to_post1_a_uid, post1_loop_a_uid};

    DeltaProjection pre_to_astro1{pre_uid, astro1_uid, dense_gen(pre_all, {0}, 1.0F), pre_all.size()};
    DeltaProjection post1a_to_astro1{
        post1_uid, astro1_uid, dense_gen(post1_group_a, {0}, 1.0F), post1_group_a.size()};

    AstrocytePopulation astro2_pop{astrocyte_generator(ASTRO_WEIGHT_MULTIPLIERS[1]), 1};
    knp::core::UID astro2_uid = astro2_pop.get_uid();
    astro2_pop.get_tags()[knp::neuron_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{pre_to_post1_b_uid};

    DeltaProjection pre_to_astro2{pre_uid, astro2_uid, dense_gen(pre_all, {0}, 1.0F), pre_all.size()};
    DeltaProjection post1b_to_astro2{
        post1_uid, astro2_uid, dense_gen(post1_group_b, {0}, 1.0F), post1_group_b.size()};

    AstrocytePopulation astro3_pop{astrocyte_generator(ASTRO_WEIGHT_MULTIPLIERS[2]), 1};
    knp::core::UID astro3_uid = astro3_pop.get_uid();
    astro3_pop.get_tags()[knp::neuron_traits::astrocyte_modulated_projections_tag] =
        std::vector<knp::core::UID>{pre_to_post2_uid, post2_loop_uid};

    DeltaProjection pre_to_astro3{pre_uid, astro3_uid, dense_gen(pre_all, {0}, 1.0F), pre_all.size()};
    DeltaProjection post2_to_astro3{post2_uid, astro3_uid, dense_gen(post2_all, {0}, 1.0F), post2_all.size()};

    network.add_population(std::move(astro1_pop));
    network.add_population(std::move(astro2_pop));
    network.add_population(std::move(astro3_pop));
    network.add_projection(std::move(pre_to_astro1));
    network.add_projection(std::move(post1a_to_astro1));
    network.add_projection(std::move(pre_to_astro2));
    network.add_projection(std::move(post1b_to_astro2));
    network.add_projection(std::move(pre_to_astro3));
    network.add_projection(std::move(post2_to_astro3));

    knp::core::UID in_ch;
    knp::framework::Model model(std::move(network));
    model.add_input_channel(in_ch, input_uid);

    auto input_gen = [STIM_END, STIM2_START, STIM2_END, STIM_INTERVAL](knp::core::Step step)
                          -> knp::core::messaging::SpikeData
    {
        knp::core::messaging::SpikeData spikes;
        if ((step < STIM_END || (step >= STIM2_START && step < STIM2_END)) && step % STIM_INTERVAL == 0)
        {
            for (size_t i = 0; i < PRE_SIZE; i++) spikes.push_back(i);
        }
        return spikes;
    };

    auto backend_path = std::filesystem::current_path() / "knp-cpu-single-threaded-backend";
    knp::framework::BackendLoader backend_loader;

    knp::framework::ModelExecutor model_executor(model, backend_loader.load(backend_path), {{in_ch, input_gen}});
    auto backend = model_executor.get_backend();

    // Logging: individual synapse weights of two projections (pre_to_post1_a, modulated; post1_loop_b, fixed).
    std::cout << "\nLegend for the individual synapse columns in the CSV:\n";
    std::cout << "  PreToPost1A_S0..S5 (modulated by astrocyte 1, x" << ASTRO_WEIGHT_MULTIPLIERS[0] << "): "
              << "pre0->A0, pre0->A1, pre1->A0, pre1->A1, pre2->A0, pre2->A1\n";
    std::cout << "  Post1LoopB_S0..S3 (NOT modulated, weight fixed permanently): "
              << "post1(2)->post1(2), post1(2)->post1(3), post1(3)->post1(2), post1(3)->post1(3)\n";

    std::ofstream data_file(filename);
    data_file << "Step,PreSpikeCount,Post1GroupASpikeCount,Post1GroupBSpikeCount,Post2SpikeCount,"
              << "Astro1_Ca,Astro1_Multiplier,Astro1_Activations,"
              << "Astro2_Ca,Astro2_Multiplier,Astro2_Activations,"
              << "Astro3_Ca,Astro3_Multiplier,Astro3_Activations,"
              << "PreToPost1A_S0,PreToPost1A_S1,PreToPost1A_S2,PreToPost1A_S3,PreToPost1A_S4,PreToPost1A_S5,"
              << "Post1LoopB_S0,Post1LoopB_S1,Post1LoopB_S2,Post1LoopB_S3\n";

    auto pre_spike_count_this_step = std::make_shared<size_t>(0);
    auto pre_spike_total = std::make_shared<size_t>(0);
    auto pre_per_neuron_totals = std::make_shared<std::vector<size_t>>(PRE_SIZE, 0);
    model_executor.add_observer<SpikeMessage>(
        [pre_spike_count_this_step, pre_spike_total, pre_per_neuron_totals](const std::vector<SpikeMessage> &messages)
        {
            size_t count = 0;
            for (const auto &msg : messages)
            {
                for (auto idx : msg.neuron_indexes_)
                {
                    ++count;
                    if (idx < pre_per_neuron_totals->size()) ++(*pre_per_neuron_totals)[idx];
                }
            }
            *pre_spike_count_this_step = count;
            *pre_spike_total += count;
        },
        {pre_uid});

    auto post1_group_a_count_this_step = std::make_shared<size_t>(0);
    auto post1_group_b_count_this_step = std::make_shared<size_t>(0);
    auto post1_group_a_total = std::make_shared<size_t>(0);
    auto post1_group_b_total = std::make_shared<size_t>(0);
    auto post1_per_neuron_totals = std::make_shared<std::vector<size_t>>(POST1_SIZE, 0);
    model_executor.add_observer<SpikeMessage>(
        [post1_group_a_count_this_step, post1_group_b_count_this_step, post1_group_a_total, post1_group_b_total,
         post1_per_neuron_totals](const std::vector<SpikeMessage> &messages)
        {
            size_t a_count = 0;
            size_t b_count = 0;
            for (const auto &msg : messages)
            {
                for (auto idx : msg.neuron_indexes_)
                {
                    if (idx < post1_per_neuron_totals->size()) ++(*post1_per_neuron_totals)[idx];
                    if (idx == 0 || idx == 1) ++a_count;
                    if (idx == 2 || idx == 3) ++b_count;
                }
            }
            *post1_group_a_count_this_step = a_count;
            *post1_group_b_count_this_step = b_count;
            *post1_group_a_total += a_count;
            *post1_group_b_total += b_count;
        },
        {post1_uid});

    auto post2_spike_count_this_step = std::make_shared<size_t>(0);
    auto post2_spike_total = std::make_shared<size_t>(0);
    auto post2_per_neuron_totals = std::make_shared<std::vector<size_t>>(POST2_SIZE, 0);
    model_executor.add_observer<SpikeMessage>(
        [post2_spike_count_this_step, post2_spike_total,
         post2_per_neuron_totals](const std::vector<SpikeMessage> &messages)
        {
            size_t count = 0;
            for (const auto &msg : messages)
            {
                for (auto idx : msg.neuron_indexes_)
                {
                    ++count;
                    if (idx < post2_per_neuron_totals->size()) ++(*post2_per_neuron_totals)[idx];
                }
            }
            *post2_spike_count_this_step = count;
            *post2_spike_total += count;
        },
        {post2_uid});

    // Registered last, so the counters above are already up to date for this step.
    auto step_counter = std::make_shared<size_t>(0);
    model_executor.add_observer<SpikeMessage>(
        [&data_file, backend, astro1_uid, astro2_uid, astro3_uid, pre_to_post1_a_uid, post1_loop_b_uid, step_counter,
         pre_spike_count_this_step, post1_group_a_count_this_step, post1_group_b_count_this_step,
         post2_spike_count_this_step](const std::vector<SpikeMessage> &)
        {
            const size_t step = (*step_counter)++;
            const auto astro1 = read_astrocyte_state(*backend, astro1_uid);
            const auto astro2 = read_astrocyte_state(*backend, astro2_uid);
            const auto astro3 = read_astrocyte_state(*backend, astro3_uid);
            const auto pre_to_post1_a_weights = read_projection_weights(*backend, pre_to_post1_a_uid);
            const auto post1_loop_b_weights = read_projection_weights(*backend, post1_loop_b_uid);

            data_file << step << ',' << *pre_spike_count_this_step << ',' << *post1_group_a_count_this_step << ','
                      << *post1_group_b_count_this_step << ',' << *post2_spike_count_this_step << ','
                      << astro1.calcium_ << ',' << astrocyte_multiplier(astro1) << ',' << astro1.activation_count_
                      << ',' << astro2.calcium_ << ',' << astrocyte_multiplier(astro2) << ','
                      << astro2.activation_count_ << ',' << astro3.calcium_ << ',' << astrocyte_multiplier(astro3)
                      << ',' << astro3.activation_count_;
            for (float weight : pre_to_post1_a_weights) data_file << ',' << weight;
            for (float weight : post1_loop_b_weights) data_file << ',' << weight;
            data_file << '\n';

            if (step % 100 == 0)
            {
                std::cout << "Step " << step << ": A1(Ca=" << astro1.calcium_ << ",x" << astrocyte_multiplier(astro1)
                           << ") A2(Ca=" << astro2.calcium_ << ",x" << astrocyte_multiplier(astro2)
                           << ") A3(Ca=" << astro3.calcium_ << ",x" << astrocyte_multiplier(astro3) << ")\n";
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

    model_executor.start([TOTAL_STEPS](knp::core::Step step) { return step < TOTAL_STEPS; });

    data_file.close();

    std::cout << "  Pre total spikes (all " << PRE_SIZE << " neurons): " << *pre_spike_total << "\n";
    for (size_t i = 0; i < PRE_SIZE; ++i)
        std::cout << "    pre neuron " << i << ": " << (*pre_per_neuron_totals)[i] << " spikes\n";

    std::cout << "  Post1 group A total spikes (neurons 0,1): " << *post1_group_a_total << "\n";
    std::cout << "  Post1 group B total spikes (neurons 2,3): " << *post1_group_b_total << "\n";
    for (size_t i = 0; i < POST1_SIZE; ++i)
        std::cout << "    post1 neuron " << i << ": " << (*post1_per_neuron_totals)[i] << " spikes\n";

    std::cout << "  Post2 total spikes (all " << POST2_SIZE << " neurons): " << *post2_spike_total << "\n";
    for (size_t i = 0; i < POST2_SIZE; ++i)
        std::cout << "    post2 neuron " << i << ": " << (*post2_per_neuron_totals)[i] << " spikes\n";

    std::cout << "  Data saved to: " << filename << "\n";
}

int main()
{
    try
    {
#ifdef _WIN32
        system("chcp 65001 > nul");
#endif

        std::cout << "==========================\n";
        std::cout << "   3 POPULATIONS + 3 INDEPENDENT ASTROCYTES\n";
        std::cout << "   - Astro1: watches pre + post1[0,1]; controls pre->post1[0,1] and its recurrent loop\n";
        std::cout << "   - Astro2: watches pre + post1[2,3]; controls ONLY pre->post1[2,3] (input)\n";
        std::cout << "   - Astro3: watches pre + post2[all]; controls pre->post2 and post2's recurrent loop\n";
        std::cout << "==========================\n\n";

        run_simulation("stimulation_multi_astro_output.csv");


        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
