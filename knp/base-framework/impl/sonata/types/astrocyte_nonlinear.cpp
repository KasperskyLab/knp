/**
 * @file astrocyte_nonlinear.cpp
 * @brief SONATA serialization and deserialization of astrocyte populations.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <knp/astrocyte-traits/astrocyte_nonlinear.h>
#include <knp/core/population.h>

#include <spdlog/spdlog.h>

#include <boost/lexical_cast.hpp>

#include "../csv_content.h"
#include "../highfive.h"
#include "../load_network.h"
#include "../save_network.h"
#include "astrocyte_modulation.h"
#include "saving_initialization.h"
#include "type_id_defines.h"


namespace knp::framework::sonata
{

template <>
std::string get_neuron_type_name<astrocyte_traits::AstrocyteNonlinear>()
{
    return "knp:AstrocyteNonlinear";
}


void save_static(const core::Population<astrocyte_traits::AstrocyteNonlinear> &population, HighFive::Group &group)
{
    PUT_NEURON_TO_DATASET(population, window_steps_, group);
    PUT_NEURON_TO_DATASET(population, window_stride_, group);
    PUT_NEURON_TO_DATASET(population, spike_count_max_, group);
    PUT_NEURON_TO_DATASET(population, pulse_time_scale_, group);
    PUT_NEURON_TO_DATASET(population, dt_, group);
    PUT_NEURON_TO_DATASET(population, pulse_duration_, group);
    PUT_NEURON_TO_DATASET(population, amplitude_max_, group);
    PUT_NEURON_TO_DATASET(population, amplitude_slope_, group);
    PUT_NEURON_TO_DATASET(population, spike_midpoint_, group);
    PUT_NEURON_TO_DATASET(population, rise_slope_, group);
    PUT_NEURON_TO_DATASET(population, fall_slope_, group);
    PUT_NEURON_TO_DATASET(population, rise_time_, group);
    PUT_NEURON_TO_DATASET(population, fall_time_scale_, group);
    PUT_NEURON_TO_DATASET(population, modulation_threshold_, group);
    PUT_NEURON_TO_DATASET(population, active_weight_multiplier_, group);
    PUT_NEURON_TO_DATASET(population, min_weight_, group);
}


void save_dynamic(const core::Population<astrocyte_traits::AstrocyteNonlinear> &population, HighFive::Group &group0)
{
    auto dyn_group = group0.createGroup(dynamic_subgroup_name);
    PUT_NEURON_TO_DATASET(population, calcium_, dyn_group);
    PUT_NEURON_TO_DATASET(population, window_step_, dyn_group);
    PUT_NEURON_TO_DATASET(population, window_position_, dyn_group);
    PUT_NEURON_TO_DATASET(population, window_check_step_, dyn_group);
    PUT_NEURON_TO_DATASET(population, spike_counter_, dyn_group);
    PUT_NEURON_TO_DATASET(population, total_spikes_, dyn_group);
    PUT_NEURON_TO_DATASET(population, activation_count_, dyn_group);
    PUT_NEURON_TO_DATASET(population, monitored_spikes_this_step_, dyn_group);
    PUT_NEURON_TO_DATASET(population, pulse_active_, dyn_group);
    PUT_NEURON_TO_DATASET(population, pulse_time_, dyn_group);
    PUT_NEURON_TO_DATASET(population, amplitude_, dyn_group);
    auto history_group = dyn_group.createGroup("spike_history_");
    for (size_t index = 0; index < population.size(); ++index)
    {
        if (!population[index].spike_history_.empty())
            history_group.createDataSet(std::to_string(index), population[index].spike_history_);
    }
}


template <>
void add_population_to_h5<core::Population<astrocyte_traits::AstrocyteNonlinear>>(
    HighFive::File &file_h5, const core::Population<astrocyte_traits::AstrocyteNonlinear> &population)
{
    SPDLOG_DEBUG("Saving astrocyte population to sonata...");
    auto group0 = initialize_adding_population(population, file_h5);
    save_static(population, group0);
    save_dynamic(population, group0);
    save_astrocyte_modulation(population.get_tags(), group0);
}


void load_static_parameters(
    const HighFive::Group &group0,
    std::vector<astrocyte_traits::astrocyte_parameters<astrocyte_traits::AstrocyteNonlinear>> &target)
{
    const size_t group_size = target.size();
    SPDLOG_TRACE("Loading nonlinear astrocyte parameters for {} cells.", group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, window_steps_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, window_stride_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, spike_count_max_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, pulse_time_scale_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, dt_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, pulse_duration_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, amplitude_max_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, amplitude_slope_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, spike_midpoint_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, rise_slope_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, fall_slope_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, rise_time_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, fall_time_scale_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, modulation_threshold_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, active_weight_multiplier_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, min_weight_, group0, group_size);
}


void load_dynamic_parameters(
    const HighFive::Group &group0,
    std::vector<astrocyte_traits::astrocyte_parameters<astrocyte_traits::AstrocyteNonlinear>> &target)
{
    const size_t group_size = target.size();
    SPDLOG_TRACE("Loading nonlinear astrocyte parameters for {} cells.", group_size);
    auto dyn_group = group0.getGroup(dynamic_subgroup_name);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, calcium_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, window_step_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, window_position_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, window_check_step_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, spike_counter_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, total_spikes_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, activation_count_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(
        target, astrocyte_traits::AstrocyteNonlinear, monitored_spikes_this_step_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, pulse_active_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, pulse_time_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, astrocyte_traits::AstrocyteNonlinear, amplitude_, dyn_group, group_size);
    if (dyn_group.exist("spike_history_"))
    {
        const auto history_group = dyn_group.getGroup("spike_history_");
        for (size_t index = 0; index < target.size(); ++index)
        {
            const auto name = std::to_string(index);
            if (history_group.exist(name))
                target[index].spike_history_ = history_group.getDataSet(name).read<std::vector<uint64_t>>();
        }
    }
}


template <>
core::Population<astrocyte_traits::AstrocyteNonlinear> load_population<astrocyte_traits::AstrocyteNonlinear>(
    const HighFive::Group &nodes_group, const std::string &population_name)
{
    auto group0 = nodes_group.getGroup(population_name).getGroup("0");
    const size_t group_size = nodes_group.getGroup(population_name).getDataSet("node_id").getDimensions().at(0);
    SPDLOG_DEBUG("Loading astrocyte population {} with {} cells.", population_name, group_size);

    std::vector<astrocyte_traits::astrocyte_parameters<astrocyte_traits::AstrocyteNonlinear>> target(group_size);
    load_static_parameters(group0, target);
    load_dynamic_parameters(group0, target);
    const knp::core::UID uid{boost::lexical_cast<boost::uuids::uuid>(population_name)};
    core::Population<astrocyte_traits::AstrocyteNonlinear> out_population(
        uid, [&target](size_t index) { return target[index]; }, group_size);
    load_astrocyte_modulation(group0, out_population.get_tags());
    return out_population;
}
}  // namespace knp::framework::sonata
