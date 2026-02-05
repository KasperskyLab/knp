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

#include <models/shared/network_constructor.h>
#include <models/shared/resource_from_weight.h>

#include <string>
#include <vector>

#include "settings.h"


// A list of short type names to make reading easier.
using DeltaSynapseData = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using DeltaSynapse = knp::synapse_traits::DeltaSynapse;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using BlifatPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using ResourceBlifatPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;


// Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
template <>
AnnotatedNetwork construct_network<knp::neuron_traits::BLIFATNeuron>(const ModelDescription &model_desc)
{
    AnnotatedNetwork result;

    for (size_t i = 0; i < classes_amount; ++i) result.data_.wta_borders_.push_back(neurons_per_column * (i + 1));

    for (int i = 0; i < num_subnetworks; ++i)
    {
        NetworkConstructor constructor(result);

        // Parameters for a default neuron.
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235859
        ResourceNeuronData default_neuron{{}};
        default_neuron.activation_threshold_ = default_threshold;
        ResourceNeuronData input_neuron = default_neuron;
        // Corresponds to L characteristic time 3.
        input_neuron.potential_decay_ = input_neuron_potential_decay;
        input_neuron.d_h_ = hebbian_plasticity;
        input_neuron.dopamine_plasticity_time_ = neuron_dopamine_period;
        input_neuron.synapse_sum_threshold_coefficient_ = threshold_weight_coeff;
        input_neuron.isi_max_ = isi_max;
        input_neuron.min_potential_ = min_potential;
        input_neuron.stability_change_parameter_ = stability_change_parameter;
        input_neuron.resource_drain_coefficient_ = resource_drain_coefficient;
        input_neuron.stochastic_stimulation_ = stochastic_stimulation;

        const auto &input_pop =
            constructor.add_population(input_neuron, num_input_neurons, NetworkConstructor::INPUT, true, "INPUT");
        const auto &output_pop =
            constructor.add_population(default_neuron, classes_amount, NetworkConstructor::OUTPUT, true, "OUTPUT");
        const auto &gate_pop =
            constructor.add_population(default_neuron, classes_amount, NetworkConstructor::NORMAL, false, "GATE");
        const auto &raster_pop = constructor.add_channeled_population(input_size, true);
        const auto &target_pop = constructor.add_channeled_population(classes_amount, false);

        result.data_.wta_data_.emplace_back().first.push_back(input_pop.uid_);

        ResourceSynapseParams raster_to_input_synapse;
        raster_to_input_synapse.rule_.synaptic_resource_ =
            resource_from_weight(base_weight_value, min_synaptic_weight, max_synaptic_weight);
        raster_to_input_synapse.rule_.dopamine_plasticity_period_ = synapse_dopamine_period;
        raster_to_input_synapse.rule_.w_min_ = min_synaptic_weight;
        raster_to_input_synapse.rule_.w_max_ = max_synaptic_weight;

        auto raster_to_input_proj = constructor.add_projection(
            raster_to_input_synapse, knp::framework::projection::creators::all_to_all<ResourceSynapse>, raster_pop,
            input_pop, true, false);
        result.data_.projections_from_raster_.push_back(raster_to_input_proj);


        DeltaSynapseData target_to_input_synapse_dopamine;
        target_to_input_synapse_dopamine.weight_ = 0.18;
        target_to_input_synapse_dopamine.delay_ = 3;
        target_to_input_synapse_dopamine.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;

        auto target_to_input_proj_dopamine = constructor.add_projection(
            target_to_input_synapse_dopamine, knp::framework::projection::creators::aligned<DeltaSynapse>, target_pop,
            input_pop, false, false);
        result.data_.projections_from_classes_.push_back(target_to_input_proj_dopamine);


        DeltaSynapseData input_to_output_synapse;
        input_to_output_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
        input_to_output_synapse.weight_ = 10;

        auto input_to_output_proj = constructor.add_projection(
            input_to_output_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, input_pop, output_pop,
            false, true);
        result.data_.wta_data_[i].second.push_back(input_to_output_proj);


        DeltaSynapseData output_to_gate_synapse;
        output_to_gate_synapse.weight_ = -10;
        output_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;

        auto output_to_gate_proj = constructor.add_projection(
            output_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, output_pop, gate_pop,
            false, false);


        DeltaSynapseData target_to_gate_synapse;
        target_to_gate_synapse.weight_ = 10.f;
        target_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

        auto target_to_gate_proj = constructor.add_projection(
            target_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, target_pop, gate_pop,
            false, false);
        result.data_.projections_from_classes_.push_back(target_to_gate_proj);


        DeltaSynapseData gate_to_input_synapse;
        gate_to_input_synapse.weight_ = 10;
        gate_to_input_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

        auto gate_to_input_proj = constructor.add_projection(
            gate_to_input_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, gate_pop, input_pop,
            false, false);


        DeltaSynapseData target_to_input_synapse_excitatory;
        target_to_input_synapse_excitatory.weight_ = -30;
        target_to_input_synapse_excitatory.delay_ = 4;
        target_to_input_synapse_excitatory.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

        auto target_to_input_proj_excitatory = constructor.add_projection(
            target_to_input_synapse_excitatory, knp::framework::projection::creators::aligned<DeltaSynapse>, target_pop,
            input_pop, false, false);
        result.data_.projections_from_classes_.push_back(target_to_input_proj_excitatory);
    }

    return result;
}
