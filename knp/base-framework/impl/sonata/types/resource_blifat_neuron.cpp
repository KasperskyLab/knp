/**
 * @file resource_blifat_neuron.cpp
 * @brief Functions for loading and saving resource STDP BLIFAT neurons.
 * @kaspersky_support A. Vartenkov
 * @date 09.04.2024
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
#include <knp/core/tag.h>
#include <knp/core/uid.h>
#include <knp/framework/tags/name.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/neuron-traits/stdp_synaptic_resource_rule.h>

#include <spdlog/spdlog.h>

#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>

#include "../csv_content.h"
#include "../highfive.h"
#include "../load_network.h"
#include "../save_network.h"
#include "type_id_defines.h"


namespace knp::framework::sonata
{

template <>
std::string get_neuron_type_name<neuron_traits::SynapticResourceSTDPBLIFATNeuron>()
{
    return "knp:SynapticResourceRuleBlifatNeuron";
}


/**
 * @brief Save tags of a population to HDF5.
 * @param population population with tags.
 * @param population_group HDF5 group of the population.
 * @details Tags are saved as two arrays of the same length, `tags_keys` and `tags_values`. The `io_type` and the
 * `name` tags are saved, a tag that the population does not have is skipped. If the population has none of them,
 * the arrays are not created.
 */
void add_tags_to_h5(
    const core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
    HighFive::NodeTraits<HighFive::Group> &population_group)
{
    const core::TagMap &tags = population.get_tags();
    std::vector<std::string> tags_values;
    std::vector<std::string> tags_keys;

    if (tags.exists(core::tags::io_type_tag))
    {
        const auto io_type = tags.get_tag<core::tags::IOType>(core::tags::io_type_tag);
        if (core::tags::IOType::input == io_type)
            tags_values.emplace_back("input");
        else if (core::tags::IOType::output == io_type)
            tags_values.emplace_back("output");
        else
            throw std::runtime_error("Unknown \"io_type\" tag value.");
        tags_keys.emplace_back(core::tags::io_type_tag);
    }

    if (tags.exists(knp::framework::tags::name_tag))
    {
        tags_values.push_back(tags.get_tag<std::string>(knp::framework::tags::name_tag));
        tags_keys.emplace_back(knp::framework::tags::name_tag);
    }

    if (tags_keys.empty()) return;

    population_group.createDataSet("tags_values", tags_values);
    population_group.createDataSet("tags_keys", tags_keys);
}


template <>
void add_population_to_h5<core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>>(
    HighFive::File &file_h5, const core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population)
{
    // TODO: It would be better if such functions were generated automatically.
    SPDLOG_TRACE("Adding population {} to HDF5...", std::string(population.get_uid()));

    if (!file_h5.exist("nodes")) throw std::runtime_error("File does not contain the \"nodes\" group.");

    HighFive::Group population_group = file_h5.createGroup("nodes/" + std::string{population.get_uid()});

    std::vector<size_t> neuron_ids;
    // std::vector<int> neuron_type_ids(
    //     population.size(), get_neuron_type_id<neuron_traits::SynapticResourceSTDPBLIFATNeuron>());

    neuron_ids.reserve(population.size());

    for (size_t i = 0; i < population.size(); ++i) neuron_ids.push_back(i);

    population_group.createDataSet("node_id", neuron_ids);
    population_group.createDataSet("node_group_index", neuron_ids);
    population_group.createDataSet("node_group_id", std::vector<size_t>(population.size(), 0));
    population_group.createDataSet(
        "node_type_id",
        std::vector<size_t>(population.size(), get_neuron_type_id<neuron_traits::SynapticResourceSTDPBLIFATNeuron>()));

    add_tags_to_h5(population, population_group);
    auto group0 = population_group.createGroup("0");

    // TODO: Need to check if all parameters are the same. If not, then save them into h5.
    // Static parameters, they don't change during inference.
    PUT_NEURON_TO_DATASET(population, n_time_steps_since_last_firing_, group0);
    PUT_NEURON_TO_DATASET(population, activation_threshold_, group0);
    PUT_NEURON_TO_DATASET(population, threshold_decay_, group0);
    PUT_NEURON_TO_DATASET(population, threshold_increment_, group0);
    PUT_NEURON_TO_DATASET(population, postsynaptic_trace_, group0);
    PUT_NEURON_TO_DATASET(population, postsynaptic_trace_decay_, group0);
    PUT_NEURON_TO_DATASET(population, postsynaptic_trace_increment_, group0);
    PUT_NEURON_TO_DATASET(population, inhibitory_conductance_, group0);
    PUT_NEURON_TO_DATASET(population, inhibitory_conductance_decay_, group0);
    PUT_NEURON_TO_DATASET(population, potential_decay_, group0);
    PUT_NEURON_TO_DATASET(population, bursting_period_, group0);
    PUT_NEURON_TO_DATASET(population, reflexive_weight_, group0);
    PUT_NEURON_TO_DATASET(population, reversal_inhibitory_potential_, group0);
    PUT_NEURON_TO_DATASET(population, absolute_refractory_period_, group0);
    PUT_NEURON_TO_DATASET(population, potential_reset_value_, group0);
    PUT_NEURON_TO_DATASET(population, min_potential_, group0);
    PUT_NEURON_TO_DATASET(population, stochastic_stimulation_, group0);

    // Synaptic rule parameters.
    // TODO: Do we need to split them into static-dynamic as well? Probably not.
    PUT_NEURON_TO_DATASET(population, free_synaptic_resource_, group0);
    PUT_NEURON_TO_DATASET(population, synaptic_resource_threshold_, group0);
    PUT_NEURON_TO_DATASET(population, resource_drain_coefficient_, group0);
    PUT_NEURON_TO_DATASET(population, stability_, group0);
    PUT_NEURON_TO_DATASET(population, stability_change_parameter_, group0);
    PUT_NEURON_TO_DATASET(population, stability_change_at_isi_, group0);
    PUT_NEURON_TO_DATASET(population, isi_max_, group0);
    PUT_NEURON_TO_DATASET(population, d_h_, group0);
    PUT_NEURON_TO_DATASET(population, last_step_, group0);
    PUT_NEURON_TO_DATASET(population, last_spike_step_, group0);
    PUT_NEURON_TO_DATASET(population, first_isi_spike_, group0);
    PUT_NEURON_TO_DATASET(population, is_being_forced_, group0);
    PUT_NEURON_TO_DATASET(population, dopamine_plasticity_time_, group0);
    {
        std::vector<int> data;
        data.reserve(population.size());
        std::transform(
            population.begin(), population.end(), std::back_inserter(data),
            [](const auto &neuron) { return static_cast<int>(neuron.isi_status_); });
        group0.createDataSet("isi_status_", data);
    }

    // Dynamic parameters.
    // They describe the current neuron state. They can change at inference.
    auto dynamic_group0 = group0.createGroup("dynamics_params");
    PUT_NEURON_TO_DATASET(population, dynamic_threshold_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, potential_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, pre_impact_potential_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, bursting_phase_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, total_blocking_period_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, dopamine_value_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, additional_threshold_, dynamic_group0);
    PUT_NEURON_TO_DATASET(population, random_number_generator_state_, dynamic_group0);
}


#define LOAD_NEURONS_PARAMETER_DEF(target, parameter, h5_group, pop_size, def_neuron)             \
    do                                                                                            \
    {                                                                                             \
        const auto values = read_parameter(h5_group, #parameter, pop_size, def_neuron.parameter); \
        for (size_t i = 0; i < target.size(); ++i) target[i].parameter = values[i];               \
    } while (false)


using ResourceNeuron = neuron_traits::SynapticResourceSTDPBLIFATNeuron;
using ResourceNeuronParams = neuron_traits::neuron_parameters<ResourceNeuron>;


/**
 * @brief Load tags of a population from HDF5.
 *
 * @param nodes_group HDF5 group of all populations.
 * @param population_name name of the population group.
 * @param population population to add the tags to.
 *
 * @details Tags are read from the `tags_keys` and `tags_values` arrays. A population that was saved without tags has
 * no such arrays, then nothing is loaded.
 */
void load_tags_to_population(
    const HighFive::Group &nodes_group, const std::string &population_name,
    core::Population<ResourceNeuron> &population)
{
    const auto population_group = nodes_group.getGroup(population_name);
    if (!population_group.exist("tags_keys") || !population_group.exist("tags_values")) return;

    const auto tags_keys = population_group.getDataSet("tags_keys").read<std::vector<std::string>>();
    const auto tags_values = population_group.getDataSet("tags_values").read<std::vector<std::string>>();
    if (tags_keys.size() != tags_values.size())
        throw std::runtime_error("Tag keys and tag values have different sizes in population " + population_name + ".");

    for (size_t i = 0; i < tags_keys.size(); ++i)
    {
        const std::string &key = tags_keys[i];
        const std::string &value = tags_values[i];

        if (core::tags::io_type_tag != key)
        {
            population.get_tags()[key] = value;
            continue;
        }

        if ("input" == value)
            population.get_tags()[key] = core::tags::IOType::input;
        else if ("output" == value)
            population.get_tags()[key] = core::tags::IOType::output;
        else
            throw std::runtime_error(
                "Unknown \"io_type\" tag value \"" + value + "\" in population " + population_name + ".");
    }
}


template <>
core::Population<neuron_traits::SynapticResourceSTDPBLIFATNeuron>
load_population<neuron_traits::SynapticResourceSTDPBLIFATNeuron>(
    const HighFive::Group &nodes_group, const std::string &population_name)
{
    SPDLOG_DEBUG("Loading nodes for population {}...", population_name);
    auto group = nodes_group.getGroup(population_name).getGroup("0");
    const size_t group_size = nodes_group.getGroup(population_name).getDataSet("node_id").getDimensions().at(0);

    // TODO: Load default neuron from JSON file.
    ResourceNeuronParams default_params{neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron>{}};
    std::vector<ResourceNeuronParams> target(group_size, default_params);
    // BLIFAT parameters.
    LOAD_NEURONS_PARAMETER_DEF(target, n_time_steps_since_last_firing_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, activation_threshold_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, threshold_decay_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, threshold_increment_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, postsynaptic_trace_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, postsynaptic_trace_decay_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, postsynaptic_trace_increment_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, inhibitory_conductance_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, inhibitory_conductance_decay_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, potential_decay_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, bursting_period_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, reflexive_weight_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, reversal_inhibitory_potential_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, absolute_refractory_period_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, potential_reset_value_, group, group_size, default_params);
    LOAD_NEURONS_PARAMETER_DEF(target, min_potential_, group, group_size, default_params);
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
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, dynamic_threshold_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, potential_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, pre_impact_potential_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, bursting_phase_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, total_blocking_period_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, dopamine_value_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::BLIFATNeuron, additional_threshold_, dyn_group, group_size);

    const knp::core::UID uid{boost::lexical_cast<boost::uuids::uuid>(population_name)};

    auto result = core::Population<ResourceNeuron>(uid, [&target](size_t index) { return target[index]; }, group_size);

    load_tags_to_population(nodes_group, population_name, result);

    return result;
}

}  // namespace knp::framework::sonata
