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

#include <string>

#include "hyperparameters.h"
// cppcheck-suppress missingInclude
#include "models/network_constructor.h"
// cppcheck-suppress missingInclude
#include "models/resource_from_weight.h"
#include "network_functions.h"


/// Short name for delta synapse.
using DeltaSynapse = knp::synapse_traits::DeltaSynapse;
/// Short name for delta synapse parameters.
using DeltaSynapseParams = knp::synapse_traits::synapse_parameters<DeltaSynapse>;
/// Short name for delta synapse projection.
using DeltaProjection = knp::core::Projection<DeltaSynapse>;
/// Short name for STDP delta synapse.
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
/// Short name for STDP delta synapse params.
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
/// Short name for STDP AltAILIF neuron parameters.
using ResourceNeuronData =
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>;


/// Structure just to store populations.
struct NetworkPopulations
{
    /// Input population.
    const PopulationInfo &input_pop_;
    /// Output population.
    const PopulationInfo &output_pop_;
    /// Gate population. Used for training.
    const PopulationInfo &gate_pop_;
    /// Population for rasterized images.
    const PopulationInfo &raster_pop_;
    /// Population for images labels.
    const PopulationInfo &target_pop_;
};


static NetworkPopulations create_populations(NetworkConstructor &constructor)
{
    // Creating neurons.
    ResourceNeuronData default_neuron;
    default_neuron.activation_threshold_ = activation_threshold;
    ResourceNeuronData input_neuron = default_neuron;
    input_neuron.potential_leak_ = potential_leak;
    input_neuron.negative_activation_threshold_ = negative_activation_threshold;
    input_neuron.potential_reset_value_ = potential_reset_value;
    input_neuron.dopamine_plasticity_time_ = dopamine_plasticity_time;
    input_neuron.isi_max_ = isi_max;
    input_neuron.d_h_ = d_h;
    input_neuron.stability_change_parameter_ = stability_change_parameter;
    input_neuron.resource_drain_coefficient_ = resource_drain_coefficient;
    input_neuron.synapse_sum_threshold_coefficient_ = synapse_sum_threshold_coefficient;

    // Creating populations using neurons.
    const auto &input_pop = constructor.add_population(
        input_neuron, classes_amount * neurons_per_column, PopulationRole::INPUT, true, "INPUT");
    const auto &output_pop =
        constructor.add_population(default_neuron, classes_amount, PopulationRole::OUTPUT, true, "OUTPUT");
    const auto &gate_pop =
        constructor.add_population(default_neuron, classes_amount, PopulationRole::NORMAL, false, "GATE");
    const auto &raster_pop = constructor.add_channeled_population(input_size, true);
    const auto &target_pop = constructor.add_channeled_population(classes_amount, false);

    // Returning them.
    return {input_pop, output_pop, gate_pop, raster_pop, target_pop};
}


static void create_projections(
    AnnotatedNetwork &network, NetworkConstructor &constructor, const NetworkPopulations &pops)
{
    // Creating synapse and projection out of it. Multiple times.

    // Synapse creation.
    ResourceSynapseParams raster_to_input_synapse;
    raster_to_input_synapse.rule_.dopamine_plasticity_period_ = raster_to_input_synapse_dopamine_plasticity_period;
    raster_to_input_synapse.rule_.w_max_ = raster_to_input_synapse_w_max;
    raster_to_input_synapse.rule_.w_min_ = raster_to_input_synapse_w_min;
    raster_to_input_synapse.rule_.synaptic_resource_ =
        resource_from_weight(0, raster_to_input_synapse.rule_.w_min_, raster_to_input_synapse.rule_.w_max_);
    // Creating projection out of synapse.
    auto raster_to_input_proj = constructor.add_projection(
        raster_to_input_synapse, knp::framework::projection::creators::all_to_all<ResourceSynapse>, pops.raster_pop_,
        pops.input_pop_, true, false);
    // Marking created projection, rasterized image will go into it.
    network.data_.projections_from_raster_.push_back(raster_to_input_proj);

    DeltaSynapseParams target_to_input_synapse_dopamine;
    target_to_input_synapse_dopamine.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;
    target_to_input_synapse_dopamine.weight_ = 0.179376 * 1000;
    target_to_input_synapse_dopamine.delay_ = 3;
    auto target_to_input_proj_dopamine = constructor.add_projection(
        target_to_input_synapse_dopamine, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.target_pop_,
        pops.input_pop_, false, false);
    // Marking created projection, labels will go into it.
    network.data_.projections_from_classes_.push_back(target_to_input_proj_dopamine);

    DeltaSynapseParams target_to_input_synapse_excitatory;
    target_to_input_synapse_excitatory.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    target_to_input_synapse_excitatory.weight_ = -30 * 1000;
    target_to_input_synapse_excitatory.delay_ = 4;
    auto target_to_input_proj_excitatory = constructor.add_projection(
        target_to_input_synapse_excitatory, knp::framework::projection::creators::all_to_all<DeltaSynapse>,
        pops.target_pop_, pops.input_pop_, false, false);
    // Marking created projection, labels will go into it.
    network.data_.projections_from_classes_.push_back(target_to_input_proj_excitatory);

    DeltaSynapseParams target_to_gate_synapse;
    target_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    target_to_gate_synapse.weight_ = 10 * 1000;
    auto target_to_gate_proj = constructor.add_projection(
        target_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.target_pop_,
        pops.gate_pop_, false, false);
    // Marking created projection, labels will go into it.
    network.data_.projections_from_classes_.push_back(target_to_gate_proj);

    DeltaSynapseParams input_to_output_synapse;
    input_to_output_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    input_to_output_synapse.weight_ = 10.f * 1000;
    knp::core::UID input_to_output_proj = constructor.add_projection(
        input_to_output_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.input_pop_,
        pops.output_pop_, false, true);
    // Connecting created projection as receiver to WTA.
    network.data_.wta_data_.back().second.push_back(input_to_output_proj);

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


/**
 * @brief Constructing AltAI network.
 * @param model_desc Model description.
 * @return Annotated network.
 * @see [Online Help](https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801)
 */
template <>
AnnotatedNetwork construct_network<knp::neuron_traits::AltAILIF>(const ModelDescription &model_desc)
{
    AnnotatedNetwork result;
    NetworkConstructor constructor(result);

    // Creating wta borders.
    for (size_t i = 0; i < classes_amount; ++i) result.data_.wta_borders_.push_back(neurons_per_column * (i + 1));

    NetworkPopulations pops = create_populations(constructor);

    // Add input_pop as WTA sender.
    result.data_.wta_data_.emplace_back().first.push_back(pops.input_pop_.uid_);

    create_projections(result, constructor, pops);

    return result;
}
