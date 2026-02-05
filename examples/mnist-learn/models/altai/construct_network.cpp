/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support D. Postnikov
 * @date 25.09.2025
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

#include <models/shared/network_constructor.h>
#include <models/shared/resource_from_weight.h>

#include <string>

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


// Create network for MNIST.
AnnotatedNetwork construct_network_altai(const ModelDescription &model_desc)
{
    AnnotatedNetwork result;
    NetworkConstructor constructor(result);

    for (size_t i = 0; i < classes_amount; ++i) result.data_.wta_borders_.push_back(neurons_per_column * (i + 1));

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

    // threshold_excess_weight_dependent
    l_neuron.synapse_sum_threshold_coefficient_ = 0.217654;


    const auto &input_pop = constructor.add_population(
        l_neuron, classes_amount * neurons_per_column, NetworkConstructor::INPUT, true, "INPUT");
    const auto &output_pop =
        constructor.add_population(default_neuron, classes_amount, NetworkConstructor::OUTPUT, true, "OUTPUT");
    const auto &gate_pop =
        constructor.add_population(default_neuron, classes_amount, NetworkConstructor::NORMAL, false, "GATE");
    const auto &raster_pop = constructor.add_channeled_population(input_size, true);
    const auto &target_pop = constructor.add_channeled_population(classes_amount, false);


    result.data_.wta_data_.emplace_back().first.push_back(input_pop.uid_);


    ResourceSynapseParams R_to_L_synapse;
    R_to_L_synapse.rule_.dopamine_plasticity_period_ = 10;
    R_to_L_synapse.rule_.w_min_ = -0.253122 * 1000;
    R_to_L_synapse.rule_.w_max_ = 0.0923957 * 1000;
    R_to_L_synapse.rule_.synaptic_resource_ =
        resource_from_weight(0, R_to_L_synapse.rule_.w_min_, R_to_L_synapse.rule_.w_max_);


    auto raster_to_input_proj = constructor.add_projection(
        R_to_L_synapse, knp::framework::projection::creators::all_to_all<ResourceSynapse>, raster_pop, input_pop, true,
        false);
    result.data_.projections_from_raster_.push_back(raster_to_input_proj);

    DeltaSynapseParams TARGET_to_L_synapse;
    TARGET_to_L_synapse.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;
    TARGET_to_L_synapse.weight_ = 0.179376 * 1000;
    TARGET_to_L_synapse.delay_ = 3;

    auto target_to_l_proj = constructor.add_projection(
        TARGET_to_L_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, target_pop, input_pop, false,
        false);
    result.data_.projections_from_classes_.push_back(target_to_l_proj);


    DeltaSynapseParams TARGET_to_L_synapse2;
    TARGET_to_L_synapse2.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    TARGET_to_L_synapse2.weight_ = -30 * 1000;
    TARGET_to_L_synapse2.delay_ = 4;

    auto target_to_l_proj2 = constructor.add_projection(
        TARGET_to_L_synapse2, knp::framework::projection::creators::all_to_all<DeltaSynapse>, target_pop, input_pop,
        false, false);
    result.data_.projections_from_classes_.push_back(target_to_l_proj2);


    DeltaSynapseParams TARGET_to_BIAS_synapse;
    TARGET_to_BIAS_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    TARGET_to_BIAS_synapse.weight_ = 10 * 1000;

    auto target_to_gate_proj = constructor.add_projection(
        TARGET_to_BIAS_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, target_pop, gate_pop,
        false, false);
    result.data_.projections_from_classes_.push_back(target_to_gate_proj);


    DeltaSynapseParams L_to_OUT_synapse;
    L_to_OUT_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    L_to_OUT_synapse.weight_ = 10.f * 1000;

    knp::core::UID l_to_out_proj = constructor.add_projection(
        L_to_OUT_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, input_pop, output_pop, false,
        true);
    result.data_.wta_data_[0].second.push_back(l_to_out_proj);


    DeltaSynapseParams OUT_to_BIAS_synapse;
    OUT_to_BIAS_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;
    OUT_to_BIAS_synapse.weight_ = -10.f;

    constructor.add_projection(
        OUT_to_BIAS_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, output_pop, gate_pop, false,
        false);


    DeltaSynapseParams BIAS_to_L_synapse;
    BIAS_to_L_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    BIAS_to_L_synapse.weight_ = 10.f * 1000;

    constructor.add_projection(
        BIAS_to_L_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, gate_pop, input_pop, false,
        false);

    // Return created network.
    return result;
}
