/**
 * @file lif_neuron.cpp
 * @brief AltaiLIF neuron procedures.
 * @kaspersky_support An. Vartenkov
 * @date 15.05.2024
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
#include <knp/neuron-traits/lif.h>

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
std::string get_neuron_type_name<neuron_traits::LIFNeuron>()
{
    return "knp:LIFNeuron";
}


void save_static(const core::Population<neuron_traits::LIFNeuron> &population, HighFive::Group &group)
{
    SPDLOG_TRACE("Saving LIF neurons static parameters.");
    PUT_NEURON_TO_DATASET(population, activation_threshold_, group);
    PUT_NEURON_TO_DATASET(population, leak_coefficient_, group);
    PUT_NEURON_TO_DATASET(population, refract_period_, group);
}


void save_dynamic(const core::Population<neuron_traits::LIFNeuron> &population, HighFive::Group &group0)
{
    SPDLOG_TRACE("Saving LIF neurons dynamic parameters.");
    auto dyn_group = group0.createGroup(dynamic_subgroup_name);
    PUT_NEURON_TO_DATASET(population, potential_, dyn_group);
    PUT_NEURON_TO_DATASET(population, refract_counter_, dyn_group);
}


template <>
void add_population_to_h5<core::Population<neuron_traits::LIFNeuron>>(
    HighFive::File &file_h5, const core::Population<neuron_traits::LIFNeuron> &population)
{
    SPDLOG_DEBUG("Saving LIF population to sonata...");
    auto group0 = initialize_adding_population(population, file_h5);
    save_static(population, group0);
    save_dynamic(population, group0);
}


void load_static_parameters(
    const HighFive::Group &group0, std::vector<neuron_traits::neuron_parameters<neuron_traits::LIFNeuron>> &target)
{
    SPDLOG_TRACE("Loading static LIF parameters.");
    const size_t group_size = target.size();
    LOAD_NEURONS_PARAMETER(target, neuron_traits::LIFNeuron, activation_threshold_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::LIFNeuron, leak_coefficient_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::LIFNeuron, refract_period_, group0, group_size);
}


void load_dynamic_parameters(
    const HighFive::Group &group0, std::vector<neuron_traits::neuron_parameters<neuron_traits::LIFNeuron>> &target)
{
    SPDLOG_TRACE("Loading LIF neurons dynamic parameters.");
    const size_t group_size = target.size();
    auto dyn_group = group0.getGroup(dynamic_subgroup_name);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::LIFNeuron, potential_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::LIFNeuron, refract_counter_, dyn_group, group_size);
}


template <>
core::Population<neuron_traits::LIFNeuron> load_population<neuron_traits::LIFNeuron>(
    const HighFive::Group &nodes_group, const std::string &population_name)
{
    SPDLOG_DEBUG("Loading nodes...");
    auto group0 = nodes_group.getGroup(population_name).getGroup("0");
    const size_t group_size = nodes_group.getGroup(population_name).getDataSet("node_id").getDimensions().at(0);

    // TODO: Load default neuron from JSON file.
    std::vector<neuron_traits::neuron_parameters<neuron_traits::LIFNeuron>> target(group_size);
    load_static_parameters(group0, target);
    load_dynamic_parameters(group0, target);
    const knp::core::UID uid{boost::lexical_cast<boost::uuids::uuid>(population_name)};
    core::Population<neuron_traits::LIFNeuron> out_population(
        uid, [&target](size_t index) { return target[index]; }, group_size);
    return out_population;
}
}  // namespace knp::framework::sonata
