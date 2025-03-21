/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 03.12.2024
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

#include "construct_network.h"

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>


using DeltaSynapseData = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using ResourceBlifatPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseGenerator = std::function<ResourceSynapseData(size_t)>;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using ResourceNeuronGenerator = std::function<ResourceNeuronData(size_t)>;


constexpr float min_synaptic_weight = -0.7;
constexpr float max_synaptic_weight = 0.864249F;
constexpr float base_weight_value = 0.000F;
constexpr float base_resource_value = 1.267F;
constexpr int neuron_dopamine_period = 10;
constexpr int synapse_dopamine_period = 10;
constexpr float L_neuron_potential_decay = 1.0 - 1.0 / 3.0;
constexpr float dopamine_parameter = 0.042F;
constexpr float threshold_weight_coeff = 0.023817F;


float resource_from_weight(float weight, float min_weight, float max_weight)
{
    float eps = 1e-6;
    if (min_weight > max_weight) std::swap(min_weight, max_weight);
    if (weight < min_weight || weight >= max_weight - eps)
        throw std::logic_error("Weight should not be less than min_weight, more than max_weight or too close to it.");
    double diff = max_weight - min_weight;
    double over = weight - min_weight;
    return static_cast<float>(over * diff / (diff - over));
}


// A dense projection generator from a default synapse.
ResourceSynapseGenerator make_dense_generator(size_t from_size, const ResourceSynapseParams &default_synapse)
{
    ResourceSynapseGenerator synapse_generator = [from_size, default_synapse](size_t index)
    {
        size_t from_index = index % from_size;
        size_t to_index = index / from_size;
        // If you need to have synapses with different parameters, change them here.
        return ResourceSynapseData{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


// A 1 to 1 simple synapse generator.
ResourceSynapseGenerator make_1_to_1_generator(const ResourceSynapseParams &default_synapse)
{
    ResourceSynapseGenerator synapse_generator = [default_synapse](size_t index) {
        return ResourceSynapseData{default_synapse, index, index};
    };
    return synapse_generator;
}


// This is a generator of connections from one group of neurons to all neurons not in this group.
DeltaProjection::SynapseGenerator make_all_to_all_sections_generator(
    size_t section_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator generator =
        [section_size, default_synapse](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        size_t section = index / (section_size * section_size);
        size_t index_in_section = index % (section_size * section_size);
        size_t input_neuron = index_in_section % section_size;
        size_t output_neuron = index_in_section / section_size;
        if (input_neuron == output_neuron) return {};  // No link to itself.
        return DeltaProjection::Synapse{
            default_synapse, section * section_size + input_neuron, section * section_size + output_neuron};
    };
    return generator;
}


DeltaProjection::SynapseGenerator make_aligned_generator(
    size_t prepopulation_size, size_t postpopulation_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator synapse_generator =
        [prepopulation_size, postpopulation_size, default_synapse](size_t index)
    {
        size_t from_index;
        size_t pack_size;
        size_t to_index;
        if (prepopulation_size >= postpopulation_size)
        {
            from_index = index;
            pack_size = prepopulation_size / postpopulation_size;
            to_index = index / pack_size;
        }
        else
        {
            to_index = index;
            pack_size = postpopulation_size / prepopulation_size;
            from_index = index / pack_size;
        }
        return DeltaProjection::Synapse{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


DeltaProjection::SynapseGenerator make_exclusive_generator(
    size_t population_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator synapse_generator = [population_size, default_synapse](size_t index)
    {
        size_t from_index;
        size_t to_index;
        from_index = index / (population_size - 1);
        to_index = index % (population_size - 1);
        if (to_index >= from_index) ++to_index;
        return DeltaProjection::Synapse{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


template <class Neuron>
struct PopulationData
{
    size_t size_;
    knp::neuron_traits::neuron_parameters<Neuron> neuron_;
};


AnnotatedNetwork create_example_network_new(int num_compound_networks)
{
    AnnotatedNetwork result;
    enum PopIndexes
    {
        INPUT = 0,
        DOPAMINE = 1,
        OUTPUT = 2,
        GATE = 3,
    };
    constexpr float dopamine_value = dopamine_parameter;
    result.data_.wta_data.resize(num_compound_networks);
    for (int i = 0; i < num_compound_networks; ++i)
    {
        // Parameters for a default neuron.
        ResourceNeuronData default_neuron{{}};
        default_neuron.activation_threshold_ = 8.571;
        ResourceNeuronData L_neuron = default_neuron;
        L_neuron.potential_decay_ = L_neuron_potential_decay;
        L_neuron.d_h_ = -dopamine_value;
        L_neuron.dopamine_plasticity_time_ = neuron_dopamine_period;
        L_neuron.synapse_sum_threshold_coefficient_ = threshold_weight_coeff;
        L_neuron.isi_max_ = 10;

        struct PopulationRole
        {
            PopulationData<ResourceNeuron> pd;
            bool for_inference;
            bool output;
            std::string name;
        };
        auto dopamine_neuron = default_neuron;
        dopamine_neuron.total_blocking_period_ = 0;
        // Create initial neuron data for populations. There are four of them.
        std::vector<PopulationRole> pop_data{
            {{150, L_neuron}, true, false, "INPUT"},
            {{150, dopamine_neuron}, false, false, "DOPAMINE"},
            {{10, default_neuron}, true, true, "OUTPUT"},
            {{10, default_neuron}, false, false, "GATE"}};

        // Creating a population. It's usually very simple as all neurons are usually the same.
        std::vector<knp::core::UID> population_uids;
        for (auto &pop_init_data : pop_data)
        {
            // A very simple neuron generator returning a default neuron.
            auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd.neuron_; };

            knp::core::UID uid;
            result.network_.add_population(ResourceBlifatPopulation{uid, neuron_generator, pop_init_data.pd.size_});
            population_uids.push_back(uid);
            result.data_.population_names[uid] = pop_init_data.name;
            if (pop_init_data.for_inference) result.data_.inference_population_uids.insert(uid);
            if (pop_init_data.output) result.data_.output_uids.push_back(uid);
        }
        result.data_.wta_data[i].first.push_back(population_uids[INPUT]);

        // Now that we added all the populations we need, we have to connect them with projections.
        // Creating a projection is more tricky, as all the connection logic should be described in a generator.

        // Create a default synapse.
        ResourceSynapseParams default_synapse;
        auto afferent_synapse = default_synapse;
        afferent_synapse.rule_.synaptic_resource_ =
            resource_from_weight(base_weight_value, min_synaptic_weight, max_synaptic_weight);
        afferent_synapse.rule_.dopamine_plasticity_period_ = synapse_dopamine_period;
        afferent_synapse.rule_.w_min_ = min_synaptic_weight;
        afferent_synapse.rule_.w_max_ = max_synaptic_weight;
        ResourceDeltaProjection input_projection{
            knp::core::UID{false}, population_uids[INPUT], make_dense_generator(28 * 28, afferent_synapse),
            28 * 28 * 150};
        result.data_.projections_from_raster.push_back(input_projection.get_uid());

        default_synapse.weight_ = 9;
        input_projection.unlock_weights();  // Trainable
        result.network_.add_projection(input_projection);
        result.data_.inference_internal_projection.insert(input_projection.get_uid());

        // Activating projection. It sends signals from labels to dopamine population.
        const DeltaSynapseData default_activating_synapse{1, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_2{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(pop_data[INPUT].pd.size_, pop_data[DOPAMINE].pd.size_, default_activating_synapse),
            pop_data[INPUT].pd.size_};
        result.network_.add_projection(projection_2);
        result.data_.wta_data[i].second.push_back(projection_2.get_uid());

        // Dopamine projection, it goes from dopamine population to input population.
        const DeltaSynapseData default_dopamine_synapse{dopamine_value, 1, knp::synapse_traits::OutputType::DOPAMINE};
        DeltaProjection projection_3{
            population_uids[DOPAMINE], population_uids[INPUT],
            make_aligned_generator(pop_data[DOPAMINE].pd.size_, pop_data[INPUT].pd.size_, default_dopamine_synapse),
            pop_data[INPUT].pd.size_};

        result.network_.add_projection(projection_3);

        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[DOPAMINE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_3.get_uid());
        }

        // Strong excitatory projection going to output neurons.
        DeltaProjection projection_4{
            knp::core::UID{false}, population_uids[OUTPUT],
            make_aligned_generator(pop_data[INPUT].pd.size_, pop_data[OUTPUT].pd.size_, default_synapse),
            pop_data[INPUT].pd.size_};
        result.data_.wta_data[i].second.push_back(projection_4.get_uid());

        result.network_.add_projection(projection_4);
        result.data_.inference_internal_projection.insert(projection_4.get_uid());

        // Blocking 1-to-1 projection.
        const DeltaSynapseData default_blocking_synapse{-20, 1, knp::synapse_traits::OutputType::BLOCKING};
        DeltaProjection projection_5{
            population_uids[OUTPUT], population_uids[GATE],
            make_aligned_generator(pop_data[OUTPUT].pd.size_, pop_data[GATE].pd.size_, default_blocking_synapse), 10};
        result.network_.add_projection(projection_5);
        if (result.data_.inference_population_uids.find(population_uids[OUTPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[GATE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_5.get_uid());
        }

        // Strong excitatory projection going from ground truth classes.
        DeltaProjection projection_6{
            knp::core::UID{false}, population_uids[DOPAMINE],
            make_aligned_generator(10, pop_data[DOPAMINE].pd.size_, default_synapse), pop_data[DOPAMINE].pd.size_};
        result.network_.add_projection(projection_6);
        result.data_.projections_from_classes.push_back(projection_6.get_uid());

        // Strong slow excitatory projection going from ground truth classes.
        auto slow_synapse = default_synapse;
        slow_synapse.delay_ = 10;
        DeltaProjection projection_7{
            knp::core::UID{false}, population_uids[GATE],
            make_aligned_generator(10, pop_data[GATE].pd.size_, slow_synapse), pop_data[GATE].pd.size_};
        result.network_.add_projection(projection_7);
        result.data_.projections_from_classes.push_back(projection_7.get_uid());

        // Strong inhibitory projection from ground truth input.
        auto inhibitory_synapse = default_synapse;
        inhibitory_synapse.weight_ = -30;
        DeltaProjection projection_8{
            knp::core::UID{false}, population_uids[GATE], make_exclusive_generator(10, inhibitory_synapse),
            10 * (pop_data[GATE].pd.size_ - 1)};
        result.data_.projections_from_classes.push_back(projection_8.get_uid());
        result.network_.add_projection(projection_8);

        // Weak excitatory projection.
        auto weak_excitatory_synapse = default_synapse;
        weak_excitatory_synapse.weight_ = 3;
        DeltaProjection projection_9{
            population_uids[GATE], population_uids[INPUT],
            make_aligned_generator(pop_data[GATE].pd.size_, pop_data[INPUT].pd.size_, weak_excitatory_synapse),
            pop_data[INPUT].pd.size_};
        result.network_.add_projection(projection_9);
        if (result.data_.inference_population_uids.find(population_uids[INPUT]) !=
                result.data_.inference_population_uids.end() &&
            result.data_.inference_population_uids.find(population_uids[GATE]) !=
                result.data_.inference_population_uids.end())
        {
            result.data_.inference_internal_projection.insert(projection_9.get_uid());
        }
    }

    // Return created network.
    return result;
}


AnnotatedNetwork parse_network_from_sonata(const std::filesystem::path &path_to_model)
{
    AnnotatedNetwork result;
    result.network_ = knp::framework::sonata::load_network(path_to_model);
    constexpr int input_projection_size = 28 * 28 * 150;
    constexpr int output_pop_size = 10;
    std::vector<knp::core::UID> input_pop_uids;
    std::vector<knp::core::UID> wta_pop_uids;
    std::vector<knp::core::UID> output_pop_uids;
    std::vector<knp::core::UID> dop_pop_uids;
    // Projections that have an appropriate size are input projections
    for (auto proj_iter = result.network_.begin_projections(); proj_iter != result.network_.end_projections();
         ++proj_iter)
    {
        size_t proj_size = std::visit([](auto &proj) { return proj.size(); }, *proj_iter);
        knp::core::UID proj_uid = std::visit([](auto &proj) { return proj.get_uid(); }, *proj_iter);
        knp::core::UID post_uid = std::visit([](auto &proj) { return proj.get_postsynaptic(); }, *proj_iter);

        if (proj_size == input_projection_size)
        {
            result.data_.projections_from_raster.push_back(proj_uid);
            result.data_.population_names.insert({post_uid, "INPUT"});
            result.data_.inference_population_uids.insert(post_uid);
            input_pop_uids.push_back(post_uid);
        }
    }
    std::set<knp::core::UID> population_uids;
    for (auto pop_iter = result.network_.begin_populations(); pop_iter != result.network_.end_populations(); ++pop_iter)
    {
        knp::core::UID pop_uid = std::visit([](auto &pop) { return pop.get_uid(); }, *pop_iter);
        size_t pop_size = std::visit([](auto &pop) { return pop.size(); }, *pop_iter);
        // Find output candidates
        if (pop_size == output_pop_size) population_uids.insert(pop_uid);
    }

    // Check output candidates: output population has no direct contact with input
    for (auto proj_iter = result.network_.begin_projections(); proj_iter != result.network_.end_projections();
         ++proj_iter)
    {
        knp::core::UID pre_uid = std::visit([](auto &proj) { return proj.get_presynaptic(); }, *proj_iter);
        knp::core::UID post_uid = std::visit([](auto &proj) { return proj.get_postsynaptic(); }, *proj_iter);
        if (std::find(input_pop_uids.begin(), input_pop_uids.end(), pre_uid) != input_pop_uids.end())
            population_uids.erase(post_uid);
        if (std::find(input_pop_uids.begin(), input_pop_uids.end(), post_uid) != input_pop_uids.end())
            population_uids.erase(pre_uid);
    }
    // The ones left in population_uids are output populations.
    for (const auto &uid : population_uids)
    {
        result.data_.output_uids.push_back(uid);
        result.data_.population_names.insert({uid, "OUTPUT"});
    }
    return result;
}
