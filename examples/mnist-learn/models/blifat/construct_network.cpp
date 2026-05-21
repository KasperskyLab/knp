/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 03.12.2024
 * @license Apache 2.0
 * @copyright © 2024-2026 AO Kaspersky Lab
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
#include <vector>

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
/// Short name for STDP delta synapse parameters.
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
/// Short name for STDP BLIFAT neuron parameters.
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;


/// Structure to store all network populations for organized access.
struct NetworkPopulations
{
    /// Input population for processing rasterized image data.
    const PopulationInfo &input_pop_;
    /// Output population for classification results.
    const PopulationInfo &output_pop_;
    /// Gate population for WTA competition mechanisms.
    const PopulationInfo &gate_pop_;
    /// Population for rasterized image channeling.
    const PopulationInfo &raster_pop_;
    /// Population for image label processing.
    const PopulationInfo &target_pop_;
};


// Create all network populations with appropriate neuron parameters and roles for BLIFAT model.
static NetworkPopulations create_populations(NetworkConstructor &constructor)
{
    // Create neurons with BLIFAT-specific parameters.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235859
    ResourceNeuronData default_neuron{{}};
    default_neuron.activation_threshold_ = default_threshold;
    ResourceNeuronData input_neuron = default_neuron;
    input_neuron.potential_decay_ = input_neuron_potential_decay;
    input_neuron.d_h_ = hebbian_plasticity;
    input_neuron.dopamine_plasticity_time_ = neuron_dopamine_period;
    input_neuron.synapse_sum_threshold_coefficient_ = threshold_weight_coeff;
    input_neuron.isi_max_ = isi_max;
    input_neuron.min_potential_ = min_potential;
    input_neuron.stability_change_parameter_ = stability_change_parameter;
    input_neuron.resource_drain_coefficient_ = resource_drain_coefficient;
    input_neuron.stochastic_stimulation_ = stochastic_stimulation;

    // Create input population with specialized BLIFAT neuron parameters.
    const auto &input_pop =
        constructor.add_population(input_neuron, num_input_neurons, PopulationRole::INPUT, true, "INPUT");
    
    // Create output population for digit classification (one neuron per digit class).
    const auto &output_pop =
        constructor.add_population(default_neuron, classes_amount, PopulationRole::OUTPUT, true, "OUTPUT");

    // Create gate population for WTA competition mechanisms.
    const auto &gate_pop =
        constructor.add_population(default_neuron, classes_amount, PopulationRole::NORMAL, false, "GATE");

    // Create raster population for channelized image processing (28x28 = 784 neurons).
    const auto &raster_pop = constructor.add_channeled_population(input_size, true);

    // Create target population for label processing (10 neurons for 10 digit classes).
    const auto &target_pop = constructor.add_channeled_population(classes_amount, false);

    // Return organized population references for projection creation.
    return {input_pop, output_pop, gate_pop, raster_pop, target_pop};
}


// Create all synaptic projections between network populations with appropriate parameters for BLIFAT.
static void create_projections(
    AnnotatedNetwork &network, NetworkConstructor &constructor, const NetworkPopulations &pops)
{
    // Configure synapse parameters for raster to input projection with STDP plasticity.
    ResourceSynapseParams raster_to_input_synapse;
    raster_to_input_synapse.rule_.synaptic_resource_ =
        resource_from_weight(base_weight_value, min_synaptic_weight, max_synaptic_weight);
    raster_to_input_synapse.rule_.dopamine_plasticity_period_ = synapse_dopamine_period;
    raster_to_input_synapse.rule_.w_min_ = min_synaptic_weight;
    raster_to_input_synapse.rule_.w_max_ = max_synaptic_weight;

    // Create projection connecting raster population to input population.
    auto raster_to_input_proj = constructor.add_projection(
        raster_to_input_synapse, knp::framework::projection::creators::all_to_all<ResourceSynapse>, pops.raster_pop_,
        pops.input_pop_, true, false);

    // Record projection for inference and monitoring.
    network.data_.projections_from_raster_.push_back(raster_to_input_proj);

    // Configure dopamine synapse for target to input projection.
    DeltaSynapseParams target_to_input_synapse_dopamine;
    target_to_input_synapse_dopamine.weight_ = 0.18;
    target_to_input_synapse_dopamine.delay_ = 3;
    target_to_input_synapse_dopamine.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;

    // Create dopamine projection connecting target population to input population.
    auto target_to_input_proj_dopamine = constructor.add_projection(
        target_to_input_synapse_dopamine, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.target_pop_,
        pops.input_pop_, false, false);

    // Record dopamine projection for label processing.
    network.data_.projections_from_classes_.push_back(target_to_input_proj_dopamine);

    // Configure synapse for input to output projection.
    DeltaSynapseParams input_to_output_synapse;
    input_to_output_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;
    input_to_output_synapse.weight_ = 10;

    // Create projection connecting input population to output population.
    auto input_to_output_proj = constructor.add_projection(
        input_to_output_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.input_pop_,
        pops.output_pop_, false, true);

    // Connect projection as receiver to WTA mechanism.
    network.data_.wta_data_.back().second.push_back(input_to_output_proj);

    // Configure synapse for output to gate projection (blocking signal).
    DeltaSynapseParams output_to_gate_synapse;
    output_to_gate_synapse.weight_ = -10;
    output_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;

    // Create blocking projection connecting output population to gate population.
    auto output_to_gate_proj = constructor.add_projection(
        output_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.output_pop_,
        pops.gate_pop_, false, false);

        // Configure synapse for target to gate projection.
    DeltaSynapseParams target_to_gate_synapse;
    target_to_gate_synapse.weight_ = 10.f;
    target_to_gate_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

    // Create projection connecting target population to gate population.
    auto target_to_gate_proj = constructor.add_projection(
        target_to_gate_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.target_pop_,
        pops.gate_pop_, false, false);

    // Record gate projection for label processing.
    network.data_.projections_from_classes_.push_back(target_to_gate_proj);

    // Configure synapse for gate to input projection.
    DeltaSynapseParams gate_to_input_synapse;
    gate_to_input_synapse.weight_ = 10;
    gate_to_input_synapse.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

    // Create gating projection connecting gate population to input population.
    auto gate_to_input_proj = constructor.add_projection(
        gate_to_input_synapse, knp::framework::projection::creators::aligned<DeltaSynapse>, pops.gate_pop_,
        pops.input_pop_, false, false);

    // Configure excitatory synapse for target to input projection.
    DeltaSynapseParams target_to_input_synapse_excitatory;
    target_to_input_synapse_excitatory.weight_ = -30;
    target_to_input_synapse_excitatory.delay_ = 4;
    target_to_input_synapse_excitatory.output_type_ = knp::synapse_traits::OutputType::EXCITATORY;

    // Create excitatory projection connecting target population to input population.
    auto target_to_input_proj_excitatory = constructor.add_projection(
        target_to_input_synapse_excitatory, knp::framework::projection::creators::aligned<DeltaSynapse>,
        pops.target_pop_, pops.input_pop_, false, false);

    // Record excitatory projection for label processing.
    network.data_.projections_from_classes_.push_back(target_to_input_proj_excitatory);
}


// Construct BLIFAT network with multi-subnetwork architecture. Each subnetwork contains the complete population and projection structure.
template <>
AnnotatedNetwork construct_network<knp::neuron_traits::BLIFATNeuron>(const ModelDescription &model_desc)
{
    AnnotatedNetwork result;

    // Create WTA borders for each digit class.
    for (size_t i = 0; i < classes_amount; ++i) result.data_.wta_borders_.push_back(neurons_per_column * (i + 1));

    // Create multiple subnetworks for BLIFAT architecture.
    for (int i = 0; i < num_subnetworks; ++i)
    {
        NetworkConstructor constructor(result);

        NetworkPopulations pops = create_populations(constructor);

        // Add input population as WTA sender for each subnetwork.
        result.data_.wta_data_.emplace_back().first.push_back(pops.input_pop_.uid_);

        create_projections(result, constructor, pops);
    }

    return result;
}
