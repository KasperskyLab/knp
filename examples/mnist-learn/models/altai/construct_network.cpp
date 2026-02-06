/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support D. Postnikov
 * @date 25.09.2025
 * @license Apache 2.0
 * @copyright © 2025-2026 AO Kaspersky Lab
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

#include <models/network_constructor.h>
#include <models/resource_from_weight.h>

#include <string>

#include "network_functions.h"
#include "settings.h"


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


struct NetworkPopulations
{
    const PopulationInfo &input_pop_, output_pop_, gate_pop_, raster_pop_, target_pop_;
};


static NetworkPopulations create_populations(NetworkConstructor &constructor)
{
    ResourceNeuronData default_neuron;
    default_neuron.activation_threshold_ = activation_threshold;
    ResourceNeuronData input_neuron = default_neuron;
    input_neuron.potential_leak_ = potential_leak;
    input_neuron.negative_activation_threshold_ = negative_activation_threshold;
    input_neuron.potential_reset_value_ = potential_reset_value;

    input_neuron.dopamine_plasticity_time_ = dopamine_plasticity_time;
    input_neuron.isi_max_ = isi_max;
    input_neuron.d_h_ = d_h;

    // stability_resource_change_ratio
    input_neuron.stability_change_parameter_ = stability_change_parameter;

    // silent synapses
    input_neuron.resource_drain_coefficient_ = resource_drain_coefficient;

    // threshold_excess_weight_dependent
    input_neuron.synapse_sum_threshold_coefficient_ = synapse_sum_threshold_coefficient;

    const auto &input_pop = constructor.add_population(
        input_neuron, classes_amount * neurons_per_column, PopulationRole::INPUT, true, "INPUT");
    const auto &output_pop =
        constructor.add_population(default_neuron, classes_amount, PopulationRole::OUTPUT, true, "OUTPUT");
    const auto &gate_pop =
        constructor.add_population(default_neuron, classes_amount, PopulationRole::NORMAL, false, "GATE");
    const auto &raster_pop = constructor.add_channeled_population(input_size, true);
    const auto &target_pop = constructor.add_channeled_population(classes_amount, false);
    return {input_pop, output_pop, gate_pop, raster_pop, target_pop};
}


static void create_projections(
    AnnotatedNetwork &network, NetworkConstructor &constructor, const NetworkPopulations &pops)
{
    ResourceSynapseParams raster_to_input_synapse;
    raster_to_input_synapse.rule_.dopamine_plasticity_period_ = raster_to_input_synapse_dopamine_plasticity_period;
    raster_to_input_synapse.rule_.w_min_ = raster_to_input_synapse_w_max;
    raster_to_input_synapse.rule_.w_max_ = raster_to_input_synapse_w_min;
    raster_to_input_synapse.rule_.synaptic_resource_ =
        resource_from_weight(0, raster_to_input_synapse.rule_.w_min_, raster_to_input_synapse.rule_.w_max_);
    auto raster_to_input_proj = constructor.add_projection(
        raster_to_input_synapse, knp::framework::projection::creators::all_to_all<ResourceSynapse>, pops.raster_pop_,
        pops.input_pop_, true, false);
    network.data_.projections_from_raster_.push_back(raster_to_input_proj);

    DeltaSynapseParams target_to_input_synapse_dopamine;
    target_to_input_synapse_dopamine.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;
    target_to_input_synapse_dopamine.weight_ = 0.179376 * 1000;
    target_to_input_synapse_dopamine.delay_ = 3;
    auto target_to_l_proj_dopamine = constructor.add_projection(
        target_to_input_synapse_dopamine, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.target_pop_,
        pops.input_pop_, false, false);
    network.data_.projections_from_classes_.push_back(target_to_l_proj_dopamine);

    DeltaSynapseParams target_to_input_synapse_excitatory;
    target_to_input_synapse_excitatory.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    target_to_input_synapse_excitatory.weight_ = -30 * 1000;
    target_to_input_synapse_excitatory.delay_ = 4;
    auto target_to_l_proj_excitatory = constructor.add_projection(
        target_to_input_synapse_excitatory, knp::framework::projection::creators::all_to_all<DeltaSynapse>,
        pops.target_pop_, pops.input_pop_, false, false);
    network.data_.projections_from_classes_.push_back(target_to_l_proj_excitatory);

    DeltaSynapseParams target_to_gate_synapse;
    target_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    target_to_gate_synapse.weight_ = 10 * 1000;
    auto target_to_gate_proj = constructor.add_projection(
        target_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.target_pop_,
        pops.gate_pop_, false, false);
    network.data_.projections_from_classes_.push_back(target_to_gate_proj);

    DeltaSynapseParams input_to_output_synapse;
    input_to_output_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    input_to_output_synapse.weight_ = 10.f * 1000;
    knp::core::UID l_to_output_proj = constructor.add_projection(
        input_to_output_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.input_pop_,
        pops.output_pop_, false, true);
    network.data_.wta_data_[0].second.push_back(l_to_output_proj);

    DeltaSynapseParams output_to_gate_synapse;
    output_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;
    output_to_gate_synapse.weight_ = -10.f;
    constructor.add_projection(
        output_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.output_pop_,
        pops.gate_pop_, false, false);

    DeltaSynapseParams gate_to_input_synapse;
    gate_to_input_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    gate_to_input_synapse.weight_ = 10.f * 1000;
    constructor.add_projection(
        gate_to_input_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.gate_pop_,
        pops.input_pop_, false, false);
}


template <>
AnnotatedNetwork construct_network<knp::neuron_traits::AltAILIF>(const ModelDescription &model_desc)
{
    AnnotatedNetwork result;
    NetworkConstructor constructor(result);

    for (size_t i = 0; i < classes_amount; ++i) result.data_.wta_borders_.push_back(neurons_per_column * (i + 1));

    NetworkPopulations pops = create_populations(constructor);

    // Add input_pop as WTA sender.
    result.data_.wta_data_.emplace_back().first.push_back(pops.input_pop_.uid_);

    create_projections(result, constructor, pops);

    return result;
}
