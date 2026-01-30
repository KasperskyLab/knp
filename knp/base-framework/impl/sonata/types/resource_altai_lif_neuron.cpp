/**
 * @file resource_altai_lif_neuron.cpp
 * @brief Functions for loading and saving resource STDP AltAILIF neurons.
 * @kaspersky_support D. Postnikov
 * @date 03.09.2025
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

#include <knp/core/population.h>
#include <knp/core/uid.h>
#include <knp/neuron-traits/altai_lif.h>
#include <knp/neuron-traits/stdp_synaptic_resource_rule.h>

#include <spdlog/spdlog.h>

#include <boost/lexical_cast.hpp>

#include "../csv_content.h"
#include "../highfive.h"
#include "../load_network.h"
#include "../save_network.h"
#include "type_id_defines.h"


namespace knp::framework::sonata
{

template <>
std::string get_neuron_type_name<neuron_traits::SynapticResourceSTDPAltAILIFNeuron>()
{
    return "knp:SynapticResourceRuleAltAILIFNeuron";
}

template <>
void add_population_to_h5<core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>>(
    HighFive::File &file_h5, const core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron> &population)
{
    // TODO: It would be better if such functions were generated automatically.
    SPDLOG_TRACE("Adding population {} to HDF5...", std::string(population.get_uid()));

    if (!file_h5.exist("nodes")) throw std::runtime_error("File does not contain the \"nodes\" group.");

    HighFive::Group population_group = file_h5.createGroup("nodes/" + std::string{population.get_uid()});

    std::vector<size_t> neuron_ids;

    neuron_ids.reserve(population.size());

    for (size_t i = 0; i < population.size(); ++i) neuron_ids.push_back(i);

    population_group.createDataSet("node_id", neuron_ids);
    population_group.createDataSet("node_group_index", neuron_ids);
    population_group.createDataSet("node_group_id", std::vector<size_t>(population.size(), 0));
    population_group.createDataSet(
        "node_type_id",
        std::vector<size_t>(
            population.size(), get_neuron_type_id<neuron_traits::SynapticResourceSTDPAltAILIFNeuron>()));
    auto group = population_group.createGroup("0");

    // TODO: Need to check if all parameters are the same. If not, then save them into h5.
    // Static parameters, they don't change during inference.
    PUT_NEURON_TO_DATASET(population, is_diff_, group);
    PUT_NEURON_TO_DATASET(population, is_reset_, group);
    PUT_NEURON_TO_DATASET(population, leak_rev_, group);
    PUT_NEURON_TO_DATASET(population, saturate_, group);
    PUT_NEURON_TO_DATASET(population, do_not_save_, group);
    PUT_NEURON_TO_DATASET(population, activation_threshold_, group);
    PUT_NEURON_TO_DATASET(population, negative_activation_threshold_, group);
    PUT_NEURON_TO_DATASET(population, potential_leak_, group);
    PUT_NEURON_TO_DATASET(population, potential_reset_value_, group);
    // Synaptic rule parameters.
    // TODO: Do we need to split them into static-dynamic as well? Probably not.
    PUT_NEURON_TO_DATASET(population, free_synaptic_resource_, group);
    PUT_NEURON_TO_DATASET(population, synaptic_resource_threshold_, group);
    PUT_NEURON_TO_DATASET(population, resource_drain_coefficient_, group);
    PUT_NEURON_TO_DATASET(population, stability_, group);
    PUT_NEURON_TO_DATASET(population, stability_change_parameter_, group);
    PUT_NEURON_TO_DATASET(population, stability_change_at_isi_, group);
    PUT_NEURON_TO_DATASET(population, isi_max_, group);
    PUT_NEURON_TO_DATASET(population, d_h_, group);
    PUT_NEURON_TO_DATASET(population, last_step_, group);
    PUT_NEURON_TO_DATASET(population, last_spike_step_, group);
    PUT_NEURON_TO_DATASET(population, first_isi_spike_, group);
    PUT_NEURON_TO_DATASET(population, is_being_forced_, group);
    PUT_NEURON_TO_DATASET(population, dopamine_plasticity_time_, group);
    {
        std::vector<int> data;
        data.reserve(population.size());
        std::transform(
            population.begin(), population.end(), std::back_inserter(data),
            [](const auto &neuron) { return static_cast<int>(neuron.isi_status_); });
        group.createDataSet("isi_status_", data);
    }

    // Dynamic parameters.
    // They describe the current neuron state. They can change at inference.
    auto dynamic_group = group.createGroup("dynamics_params");
    PUT_NEURON_TO_DATASET(population, dopamine_value_, dynamic_group);
    PUT_NEURON_TO_DATASET(population, additional_threshold_, dynamic_group);
}


#define LOAD_NEURONS_PARAMETER_DEF(target, parameter, h5_group, pop_size, def_neuron)             \
    do                                                                                            \
    {                                                                                             \
        const auto values = read_parameter(h5_group, #parameter, pop_size, def_neuron.parameter); \
        for (size_t i = 0; i < target.size(); ++i) target[i].parameter = values[i];               \
    } while (false)


using ResourceNeuron = neuron_traits::SynapticResourceSTDPAltAILIFNeuron;
using ResourceNeuronParams = neuron_traits::neuron_parameters<ResourceNeuron>;


template <>
core::Population<neuron_traits::SynapticResourceSTDPAltAILIFNeuron>
load_population<neuron_traits::SynapticResourceSTDPAltAILIFNeuron>(
    const HighFive::Group &nodes_group, const std::string &population_name)
{
    SPDLOG_DEBUG("Loading nodes for population {}...", population_name);
    auto group = nodes_group.getGroup(population_name).getGroup("0");
    const size_t group_size = nodes_group.getGroup(population_name).getDataSet("node_id").getDimensions().at(0);

    // TODO: Load default neuron from JSON file.
    ResourceNeuronParams default_params{neuron_traits::neuron_parameters<neuron_traits::AltAILIF>{}};
    std::vector<ResourceNeuronParams> target(group_size, default_params);
    // AltAILIF parameters.
    LOAD_NEURONS_PARAMETER_DEF(target, is_diff_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, is_reset_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, leak_rev_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, saturate_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, do_not_save_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, activation_threshold_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, negative_activation_threshold_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, potential_leak_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, potential_reset_value_, group, group_size, default_params);
    // Synaptic rule parameters.
    LOAD_NEURONS_PARAMETER_DEF(target, free_synaptic_resource_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, synaptic_resource_threshold_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, resource_drain_coefficient_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, stability_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, stability_change_parameter_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, stability_change_at_isi_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, isi_max_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, d_h_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, last_step_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, last_spike_step_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, first_isi_spike_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, is_being_forced_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, dopamine_plasticity_time_, group, group_size, default_params);
    {
        const auto values =
            read_parameter(group, "isi_status_", group_size, static_cast<int>(default_params.isi_status_));
        for (size_t i = 0; i < target.size(); ++i)
        {
            target[i].isi_status_ = static_cast<neuron_traits::ISIPeriodType>(values[i]);
        }
    }

    // Dynamic parameters.
    auto dyn_group = group.getGroup("dynamics_params");
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AltAILIF, dopamine_value_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AltAILIF, additional_threshold_, dyn_group, group_size);

    const knp::core::UID uid{boost::lexical_cast<boost::uuids::uuid>(population_name)};
    return core::Population<ResourceNeuron>(
        uid, [&target](size_t index) { return target[index]; }, group_size);
}

}  // namespace knp::framework::sonata
