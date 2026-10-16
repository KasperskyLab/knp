#include <knp/core/population.h>
#include <knp/neuron-traits/astrocyte.h>

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
std::string get_neuron_type_name<neuron_traits::AstrocyteCell>()
{
    return "knp:AstrocyteCell";
}


void save_static(const core::Population<neuron_traits::AstrocyteCell> &population, HighFive::Group &group)
{
    SPDLOG_TRACE("Saving astrocyte static parameters.");
    PUT_NEURON_TO_DATASET(population, calcium_max_, group);
    PUT_NEURON_TO_DATASET(population, calcium_step_, group);
    PUT_NEURON_TO_DATASET(population, modulation_threshold_, group);
    PUT_NEURON_TO_DATASET(population, spike_threshold_, group);
    PUT_NEURON_TO_DATASET(population, base_weight_, group);
    PUT_NEURON_TO_DATASET(population, active_weight_multiplier_, group);
    PUT_NEURON_TO_DATASET(population, min_weight_, group);
    PUT_NEURON_TO_DATASET(population, max_weight_, group);
}


void save_dynamic(const core::Population<neuron_traits::AstrocyteCell> &population, HighFive::Group &group0)
{
    SPDLOG_TRACE("Saving astrocyte dynamic parameters.");
    auto dyn_group = group0.createGroup(dynamic_subgroup_name);
    PUT_NEURON_TO_DATASET(population, calcium_, dyn_group);
    PUT_NEURON_TO_DATASET(population, rising_, dyn_group);
    PUT_NEURON_TO_DATASET(population, spike_counter_, dyn_group);
    PUT_NEURON_TO_DATASET(population, total_spikes_, dyn_group);
    PUT_NEURON_TO_DATASET(population, activation_count_, dyn_group);
    PUT_NEURON_TO_DATASET(population, monitored_spikes_this_step_, dyn_group);
}


template <>
void add_population_to_h5<core::Population<neuron_traits::AstrocyteCell>>(
    HighFive::File &file_h5, const core::Population<neuron_traits::AstrocyteCell> &population)
{
    SPDLOG_DEBUG("Saving astrocyte population to sonata...");
    auto group0 = initialize_adding_population(population, file_h5);
    save_static(population, group0);
    save_dynamic(population, group0);
}


void load_static_parameters(
    const HighFive::Group &group0,
    std::vector<neuron_traits::neuron_parameters<neuron_traits::AstrocyteCell>> &target)
{
    SPDLOG_TRACE("Loading static astrocyte parameters.");
    const size_t group_size = target.size();
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, calcium_max_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, calcium_step_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, modulation_threshold_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, spike_threshold_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, base_weight_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, active_weight_multiplier_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, min_weight_, group0, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, max_weight_, group0, group_size);
}


void load_dynamic_parameters(
    const HighFive::Group &group0,
    std::vector<neuron_traits::neuron_parameters<neuron_traits::AstrocyteCell>> &target)
{
    SPDLOG_TRACE("Loading astrocyte dynamic parameters.");
    const size_t group_size = target.size();
    auto dyn_group = group0.getGroup(dynamic_subgroup_name);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, calcium_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, rising_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, spike_counter_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, total_spikes_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(target, neuron_traits::AstrocyteCell, activation_count_, dyn_group, group_size);
    LOAD_NEURONS_PARAMETER(
        target, neuron_traits::AstrocyteCell, monitored_spikes_this_step_, dyn_group, group_size);
}


template <>
core::Population<neuron_traits::AstrocyteCell> load_population<neuron_traits::AstrocyteCell>(
    const HighFive::Group &nodes_group, const std::string &population_name)
{
    SPDLOG_DEBUG("Loading nodes...");
    auto group0 = nodes_group.getGroup(population_name).getGroup("0");
    const size_t group_size = nodes_group.getGroup(population_name).getDataSet("node_id").getDimensions().at(0);

    std::vector<neuron_traits::neuron_parameters<neuron_traits::AstrocyteCell>> target(group_size);
    load_static_parameters(group0, target);
    load_dynamic_parameters(group0, target);
    const knp::core::UID uid{boost::lexical_cast<boost::uuids::uuid>(population_name)};
    core::Population<neuron_traits::AstrocyteCell> out_population(
        uid, [&target](size_t index) { return target[index]; }, group_size);
    return out_population;
}
}  // namespace knp::framework::sonata
