/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 03.12.2024
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

#include "construct_network.h"

#include "shared.h"


// A list of short type names to make reading easier.
using DeltaSynapseData = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using ResourceBlifatPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;


// Intermediate population neurons.
template <class Neuron>
struct PopulationData
{
    size_t size_;
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235859
    knp::neuron_traits::neuron_parameters<Neuron> neuron_;
};


enum PopIndexes
{
    INPUT = 0,
    DOPAMINE = 1,
    OUTPUT = 2,
    GATE = 3
};


// Calculate synaptic resource value given synapse weight.
float resource_from_weight(float weight, float min_weight, float max_weight)
{
    // Max weight is only possible with infinite resource, so we should select a value less than that.
    float eps = 1e-6;
    if (min_weight > max_weight) std::swap(min_weight, max_weight);
    if (weight < min_weight || weight >= max_weight - eps)
        throw std::logic_error("Weight should not be less than min_weight, more than max_weight or too close to it.");
    double diff = max_weight - min_weight;
    double over = weight - min_weight;
    return static_cast<float>(over * diff / (diff - over));
}


// Add populations to the network.
auto add_subnetwork_populations(AnnotatedNetwork &result)
{
    // Parameters for a default neuron.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235859
    ResourceNeuronData default_neuron{{}};
    default_neuron.activation_threshold_ = default_threshold;
    ResourceNeuronData l_neuron = default_neuron;
    // Corresponds to L characteristic time 3.
    l_neuron.potential_decay_ = l_neuron_potential_decay;
    l_neuron.d_h_ = hebbian_plasticity;
    l_neuron.dopamine_plasticity_time_ = neuron_dopamine_period;
    l_neuron.synapse_sum_threshold_coefficient_ = threshold_weight_coeff;
    l_neuron.isi_max_ = 10;
    l_neuron.min_potential_ = 0;
    l_neuron.stability_change_parameter_ = 0.05F;
    l_neuron.resource_drain_coefficient_ = 27;
    l_neuron.stochastic_stimulation_ = 2.212;

    struct PopulationRole
    {
        PopulationData<ResourceNeuron> pd_;
        bool for_inference_;
        bool output_;
        std::string name_;
    };
    auto dopamine_neuron = default_neuron;
    dopamine_neuron.total_blocking_period_ = 0;
    // Create initial neuron data for populations. There are four of them.
    std::vector<PopulationRole> pop_data{
        {{num_input_neurons, l_neuron}, true, false, "INPUT"},
        {{num_possible_labels, default_neuron}, true, true, "OUTPUT"},
        {{num_possible_labels, default_neuron}, false, false, "GATE"}};

    // Creating a population. It's usually very simple as all neurons are usually the same.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
    std::vector<knp::core::UID> population_uids;
    for (auto &pop_init_data : pop_data)
    {
        // A very simple neuron generator returning a default neuron.
        auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd_.neuron_; };

        knp::core::UID uid;
        result.network_.add_population(ResourceBlifatPopulation{uid, neuron_generator, pop_init_data.pd_.size_});
        population_uids.push_back(uid);
        result.data_.population_names_[uid] = pop_init_data.name_;
        if (pop_init_data.for_inference_) result.data_.inference_population_uids_.insert(uid);
        if (pop_init_data.output_) result.data_.output_uids_.push_back(uid);
    }

    result.data_.wta_data_.emplace_back().first.push_back(population_uids[INPUT]);
    return std::make_pair(population_uids, pop_data);
}

// Create network for MNIST.
// Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
AnnotatedNetwork create_example_network(int num_compound_networks)
{
    AnnotatedNetwork result;
    for (int i = 0; i < num_compound_networks; ++i)
    {
        auto [population_uids, pop_data] = add_subnetwork_populations(result);

        ResourceSynapseParams R_to_INPUT_synapse;
        R_to_INPUT_synapse.rule_.synaptic_resource_ =
            resource_from_weight(base_weight_value, min_synaptic_weight, max_synaptic_weight);
        R_to_INPUT_synapse.rule_.dopamine_plasticity_period_ = synapse_dopamine_period;
        R_to_INPUT_synapse.rule_.w_min_ = min_synaptic_weight;
        R_to_INPUT_synapse.rule_.w_max_ = max_synaptic_weight;

        ResourceDeltaProjection R_to_INPUT_projection =
            knp::framework::projection::creators::all_to_all<ResourceSynapse>(
                knp::core::UID{false}, population_uids[INPUT], input_size, num_input_neurons,
                [&R_to_INPUT_synapse](size_t, size_t) { return R_to_INPUT_synapse; });
        result.data_.projections_from_raster_.push_back(R_to_INPUT_projection.get_uid());
        R_to_INPUT_projection.unlock_weights();  // Trainable
        result.network_.add_projection(R_to_INPUT_projection);
        result.data_.inference_internal_projection_.insert(R_to_INPUT_projection.get_uid());


        DeltaSynapseData TARGET_to_INPUT_synapse;
        TARGET_to_INPUT_synapse.weight_ = 0.18;
        TARGET_to_INPUT_synapse.delay_ = 3;
        TARGET_to_INPUT_synapse.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;

        DeltaProjection TARGET_to_INPUT_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID{false}, population_uids[INPUT], num_possible_labels, pop_data[INPUT].pd_.size_,
                [&TARGET_to_INPUT_synapse](size_t, size_t) { return TARGET_to_INPUT_synapse; });
        result.network_.add_projection(TARGET_to_INPUT_projection);
        result.data_.projections_from_classes_.push_back(TARGET_to_INPUT_projection.get_uid());


        DeltaSynapseData INPUT_to_OUTPUT_synapse;
        INPUT_to_OUTPUT_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        INPUT_to_OUTPUT_synapse.weight_ = 10;

        DeltaProjection INPUT_to_OUTPUT_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[INPUT], population_uids[OUTPUT], pop_data[INPUT].pd_.size_, pop_data[OUTPUT].pd_.size_,
                [&INPUT_to_OUTPUT_synapse](size_t, size_t) { return INPUT_to_OUTPUT_synapse; });
        result.data_.wta_data_[i].second.push_back(INPUT_to_OUTPUT_projection.get_uid());
        result.network_.add_projection(INPUT_to_OUTPUT_projection);
        result.data_.inference_internal_projection_.insert(INPUT_to_OUTPUT_projection.get_uid());


        DeltaSynapseData OUTPUT_to_GATE_synapse;
        OUTPUT_to_GATE_synapse.weight_ = -10;
        OUTPUT_to_GATE_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;

        DeltaProjection OUTPUT_to_GATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[OUTPUT], population_uids[GATE], pop_data[OUTPUT].pd_.size_, pop_data[GATE].pd_.size_,
                [&OUTPUT_to_GATE_synapse](size_t, size_t) { return OUTPUT_to_GATE_synapse; });
        result.network_.add_projection(OUTPUT_to_GATE_projection);


        DeltaSynapseData TARGET_to_GATE_synapse;
        TARGET_to_GATE_synapse.weight_ = 10;
        TARGET_to_GATE_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

        DeltaProjection TARGET_to_GATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID{false}, population_uids[GATE], num_possible_labels, pop_data[GATE].pd_.size_,
                [&TARGET_to_GATE_synapse](size_t, size_t) { return TARGET_to_GATE_synapse; });
        result.network_.add_projection(TARGET_to_GATE_projection);
        result.data_.projections_from_classes_.push_back(TARGET_to_GATE_projection.get_uid());


        DeltaSynapseData GATE_to_INPUT_synapse;
        GATE_to_INPUT_synapse.weight_ = 10;
        GATE_to_INPUT_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

        DeltaProjection GATE_to_INPUT_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[GATE], population_uids[INPUT], pop_data[GATE].pd_.size_, pop_data[INPUT].pd_.size_,
                [&GATE_to_INPUT_synapse](size_t, size_t) { return GATE_to_INPUT_synapse; });
        result.network_.add_projection(GATE_to_INPUT_projection);

        DeltaSynapseData TARGET_to_INPUT_synapse2;
        TARGET_to_INPUT_synapse2.weight_ = -30;
        TARGET_to_INPUT_synapse2.delay_ = 4;
        TARGET_to_INPUT_synapse2.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        DeltaProjection TARGET_to_INPUT_projection2 =
            knp::framework::projection::creators::all_to_all<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID{false}, population_uids[INPUT], num_possible_labels, num_input_neurons,
                [&TARGET_to_INPUT_synapse2](size_t, size_t) { return TARGET_to_INPUT_synapse2; });
        result.network_.add_projection(TARGET_to_INPUT_projection2);
        result.data_.projections_from_classes_.push_back(TARGET_to_INPUT_projection2.get_uid());
    }

    // Return created network.
    return result;
}
