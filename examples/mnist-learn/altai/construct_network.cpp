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

#include "shared_network.h"


// A list of short type names to make reading easier.
using DeltaSynapseParams = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using DeltaSynapse = knp::synapse_traits::DeltaSynapse;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using AltAILIFPopulation = knp::core::Population<knp::neuron_traits::AltAILIF>;
using ResourceAltAILIFPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;


// Intermediate population neurons.
template <class Neuron>
struct PopulationData
{
    size_t size_;
    knp::neuron_traits::neuron_parameters<Neuron> neuron_;
};


enum PopIndexes
{
    L,
    OUT,
    BIAS
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
    ResourceNeuronData default_neuron;
    default_neuron.activation_threshold_ = 8531;
    ResourceNeuronData l_neuron = default_neuron;
    l_neuron.potential_leak_ = static_cast<uint16_t>(-(1.f - 1.f / 3.f) * 1000);
    l_neuron.negative_activation_threshold_ = 0;
    l_neuron.potential_reset_value_ = 0;

    l_neuron.dopamine_plasticity_time_ = 10;
    l_neuron.isi_max_ = 10;
    l_neuron.d_h_ = -0.1765261f * 1000;

    // stability_resource_change_ratio
    l_neuron.stability_change_parameter_ = 0.0497573 / 1000;

    // silent synapses
    l_neuron.resource_drain_coefficient_ = 27;

    // chartime * hebbian_plasticity_chartime_ratio
    // l_neuron.dopamine_plasticity_time_ = 3 * 2.72f;

    // threshold_excess_weight_dependent
    l_neuron.synapse_sum_threshold_coefficient_ = 0.217654;

    struct PopulationRole
    {
        PopulationData<ResourceNeuron> pd_;
        bool for_inference_;
        bool output_;
        std::string name_;
    };

    //
    std::vector<PopulationRole> pop_data{
        {{classes_amount * neurons_per_column, l_neuron}, true, false, "L"},
        {{classes_amount, default_neuron}, true, true, "OUT"},
        {{classes_amount, default_neuron}, false, false, "BIAS"}};

    std::vector<knp::core::UID> population_uids;
    for (auto &pop_init_data : pop_data)
    {
        // A very simple neuron generator returning a default neuron.
        auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd_.neuron_; };

        knp::core::UID uid;
        result.network_.add_population(ResourceAltAILIFPopulation(uid, neuron_generator, pop_init_data.pd_.size_));
        population_uids.push_back(uid);
        result.data_.population_names_[uid] = pop_init_data.name_;
        if (pop_init_data.for_inference_) result.data_.inference_population_uids_.insert(uid);
        if (pop_init_data.output_) result.data_.output_uids_.push_back(uid);
    }

    result.data_.wta_data_.emplace_back().first.push_back(population_uids[L]);
    return std::make_pair(population_uids, pop_data);
}

// Create network for MNIST.
AnnotatedNetwork create_example_network(int num_compound_networks)
{
    AnnotatedNetwork result;
    for (int i = 0; i < num_compound_networks; ++i)
    {
        auto [population_uids, pop_data] = add_subnetwork_populations(result);

        ResourceSynapseParams R_to_L_synapse;
        //-weight_inc
        // R_to_L_synapse.rule_.d_u_ = 0.277539;
        // w_min and w_max * 1000 because of 8531
        R_to_L_synapse.rule_.dopamine_plasticity_period_ = 10;
        R_to_L_synapse.rule_.w_min_ = -0.253122 * 1000;
        R_to_L_synapse.rule_.w_max_ = 0.0923957 * 1000;
        R_to_L_synapse.rule_.synaptic_resource_ =
            resource_from_weight(0, R_to_L_synapse.rule_.w_min_, R_to_L_synapse.rule_.w_max_);


        ResourceDeltaProjection R_to_L_projection = knp::framework::projection::creators::all_to_all<ResourceSynapse>(
            knp::core::UID(false), population_uids[L], input_size, pop_data[L].pd_.size_,
            [&R_to_L_synapse](size_t, size_t) { return R_to_L_synapse; });
        result.data_.projections_from_raster_.push_back(R_to_L_projection.get_uid());
        R_to_L_projection.unlock_weights();  // Trainable
        result.network_.add_projection(R_to_L_projection);
        result.data_.inference_internal_projection_.insert(R_to_L_projection.get_uid());

        DeltaSynapseParams TARGET_to_L_synapse;
        TARGET_to_L_synapse.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;
        TARGET_to_L_synapse.weight_ = 0.179376 * 1000;
        TARGET_to_L_synapse.delay_ = 3;

        DeltaProjection TARGET_to_L_projection = knp::framework::projection::creators::aligned<DeltaSynapse>(
            knp::core::UID(false), population_uids[L], classes_amount, pop_data[L].pd_.size_,
            [&TARGET_to_L_synapse](size_t, size_t) { return TARGET_to_L_synapse; });
        result.network_.add_projection(TARGET_to_L_projection);
        result.data_.projections_from_classes_.push_back(TARGET_to_L_projection.get_uid());


        DeltaSynapseParams TARGET_to_L_synapse2;
        TARGET_to_L_synapse2.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        TARGET_to_L_synapse2.weight_ = -30 * 1000;
        TARGET_to_L_synapse2.delay_ = 4;

        DeltaProjection TARGET_to_L_projection2 = knp::framework::projection::creators::all_to_all<DeltaSynapse>(
            knp::core::UID(false), population_uids[L], classes_amount, pop_data[L].pd_.size_,
            [&TARGET_to_L_synapse2](size_t, size_t) { return TARGET_to_L_synapse2; });
        result.network_.add_projection(TARGET_to_L_projection2);
        result.data_.projections_from_classes_.push_back(TARGET_to_L_projection2.get_uid());


        DeltaSynapseParams TARGET_to_BIAS_synapse;
        TARGET_to_BIAS_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        TARGET_to_BIAS_synapse.weight_ = 10 * 1000;

        DeltaProjection TARGET_to_BIAS_projection = knp::framework::projection::creators::aligned<DeltaSynapse>(
            knp::core::UID(false), population_uids[BIAS], classes_amount, pop_data[BIAS].pd_.size_,
            [&TARGET_to_BIAS_synapse](size_t, size_t) { return TARGET_to_BIAS_synapse; });
        result.network_.add_projection(TARGET_to_BIAS_projection);
        result.data_.projections_from_classes_.push_back(TARGET_to_BIAS_projection.get_uid());


        DeltaSynapseParams L_to_OUT_synapse;
        L_to_OUT_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        L_to_OUT_synapse.weight_ = 10.f * 1000;

        DeltaProjection L_to_OUT_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID(false), population_uids[OUT], pop_data[L].pd_.size_, pop_data[OUT].pd_.size_,
                [&L_to_OUT_synapse](size_t, size_t) { return L_to_OUT_synapse; });
        result.network_.add_projection(L_to_OUT_projection);
        result.data_.inference_internal_projection_.insert(L_to_OUT_projection.get_uid());
        result.data_.wta_data_[i].second.push_back(L_to_OUT_projection.get_uid());


        DeltaSynapseParams OUT_to_BIAS_synapse;
        OUT_to_BIAS_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;
        OUT_to_BIAS_synapse.weight_ = -10.f;

        DeltaProjection OUT_to_BIAS_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[OUT], population_uids[BIAS], pop_data[OUT].pd_.size_, pop_data[BIAS].pd_.size_,
                [&OUT_to_BIAS_synapse](size_t, size_t) { return OUT_to_BIAS_synapse; });
        result.network_.add_projection(OUT_to_BIAS_projection);


        DeltaSynapseParams BIAS_to_L_synapse;
        BIAS_to_L_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        BIAS_to_L_synapse.weight_ = 10.f * 1000;

        DeltaProjection BIAS_to_L_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[BIAS], population_uids[L], pop_data[BIAS].pd_.size_, pop_data[L].pd_.size_,
                [&BIAS_to_L_synapse](size_t, size_t) { return BIAS_to_L_synapse; });
        result.network_.add_projection(BIAS_to_L_projection);
    }

    // Return created network.
    return result;
}
