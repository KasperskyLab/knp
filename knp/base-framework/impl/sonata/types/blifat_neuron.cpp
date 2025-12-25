/**
 * @file blifat_neuron.cpp
 * @brief BLIFAT neuron procedures.
 * @kaspersky_support An. Vartenkov
 * @date 28.03.2024
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

#include <knp/core/population.h>
#include <knp/core/uid.h>
#include <knp/neuron-traits/blifat.h>

#include <spdlog/spdlog.h>

#include <boost/lexical_cast.hpp>

#include "../csv_content.h"
#include "../highfive.h"
#include "../load_network.h"
#include "../save_network.h"
#include "saving_initialization.h"
#include "type_id_defines.h"


namespace knp::framework::sonata
{

template <>
std::string get_neuron_type_name<neuron_traits::BLIFATNeuron>()
{
    return "knp:BasicBlifatNeuron";
}


void save_static(const core::Population<knp::neuron_traits::BLIFATNeuron> &population, HighFive::Group &group)
{
    // TODO: Need to check if all parameters are the same. If not, save them into h5.
    // Static.
    PUT_NEURON_TO_DATASET(population, n_time_steps_since_last_firing_, group);
    PUT_NEURON_TO_DATASET(population, activation_threshold_, group);
    PUT_NEURON_TO_DATASET(population, threshold_decay_, group);
    PUT_NEURON_TO_DATASET(population, threshold_increment_, group);
    PUT_NEURON_TO_DATASET(population, postsynaptic_trace_, group);
    PUT_NEURON_TO_DATASET(population, postsynaptic_trace_decay_, group);
    PUT_NEURON_TO_DATASET(population, postsynaptic_trace_increment_, group);
    PUT_NEURON_TO_DATASET(population, inhibitory_conductance_, group);
    PUT_NEURON_TO_DATASET(population, inhibitory_conductance_decay_, group);
    PUT_NEURON_TO_DATASET(population, potential_decay_, group);
    PUT_NEURON_TO_DATASET(population, bursting_period_, group);
    PUT_NEURON_TO_DATASET(population, reflexive_weight_, group);
    PUT_NEURON_TO_DATASET(population, reversal_inhibitory_potential_, group);
    PUT_NEURON_TO_DATASET(population, absolute_refractory_period_, group);
    PUT_NEURON_TO_DATASET(population, potential_reset_value_, group);
    PUT_NEURON_TO_DATASET(population, min_potential_, group);
    PUT_NEURON_TO_DATASET(population, stochastic_stimulation_, group);
}


void save_dynamic(const core::Population<knp::neuron_traits::BLIFATNeuron> &population, HighFive::Group &group0)
{
    auto dynamic_group = group0.createGroup(dynamic_subgroup_name);
    PUT_NEURON_TO_DATASET(population, dynamic_threshold_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, potential_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, pre_impact_potential_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, bursting_phase_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, total_blocking_period_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, dopamine_value_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, random_number_generator_state_, dynamic_group);
}


template <>
void add_population_to_h5<core::Population<knp::neuron_traits::BLIFATNeuron>>(
    HighFive::File &file_h5, const core::Population<knp::neuron_traits::BLIFATNeuron> &population)
{
    SPDLOG_DEBUG("Saving BLIFAT nodes...");
    auto group0 = initialize_adding_population(population, file_h5);
    save_static(population, group0);
    save_dynamic(population, group0);
}


void load_static_parameters(
    std::vector<neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron>> &target, const HighFive::Group &group0)
{
    const size_t group_size = target.size();
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, n_time_steps_since_last_firing_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, activation_threshold_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, threshold_decay_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, threshold_increment_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, postsynaptic_trace_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, postsynaptic_trace_decay_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, postsynaptic_trace_increment_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, inhibitory_conductance_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, inhibitory_conductance_decay_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, potential_decay_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, bursting_period_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, reflexive_weight_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, reversal_inhibitory_potential_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, absolute_refractory_period_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, potential_reset_value_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, min_potential_, group0, group_size);
}


void load_dynamic_parameters(
    std::vector<neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron>> &target, const HighFive::Group &group0)
{
    const size_t group_size = target.size();
    auto dyn_group = group0.getGroup(dynamic_subgroup_name);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, dynamic_threshold_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, potential_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, pre_impact_potential_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, bursting_phase_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, total_blocking_period_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, dopamine_value_, dyn_group, group_size);
}


template <>
core::Population<neuron_traits::BLIFATNeuron> load_population<neuron_traits::BLIFATNeuron>(
    const HighFive::Group &nodes_group, const std::string &population_name)
{
    SPDLOG_DEBUG("Loading BLIFAT nodes...");
    auto group0 = nodes_group.getGroup(population_name).getGroup("0");
    const size_t group_size = nodes_group.getGroup(population_name).getDataSet("node_id").getDimensions().at(0);

    // TODO: Load default neuron from JSON file.
    std::vector<neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron>> target(group_size);
    load_static_parameters(target, group0);
    load_dynamic_parameters(target, group0);

    const knp::core::UID uid{boost::lexical_cast<boost::uuids::uuid>(population_name)};
    core::Population<neuron_traits::BLIFATNeuron> out_population(
        uid, [&target](size_t index) { return target[index]; }, group_size);
    return out_population;
}


}  // namespace knp::framework::sonata
