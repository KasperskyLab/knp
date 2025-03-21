/**
 * @file resource_delta_synapse.cpp
 * @brief Functions for loading and saving resource-based STDP-compatible delta synapses.
 * @kaspersky_support A. Vartenkov
 * @date 09.04.2024
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

#include <knp/core/projection.h>
#include <knp/synapse-traits/delta.h>
#include <knp/synapse-traits/stdp_synaptic_resource_rule.h>

#include <filesystem>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>

#include "../csv_content.h"
#include "../highfive.h"
#include "../load_network.h"
#include "../save_network.h"
#include "type_id_defines.h"


namespace knp::framework::sonata
{
namespace fs = std::filesystem;
using ResourceDeltaSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;


template <>
std::string get_synapse_type_name<ResourceDeltaSynapse>()
{
    return "knp:StdpResourceDeltaSynapse";
}


#define PUT_SYNAPSE_RULE_TO_DATASET(proj, param, group)                            \
    {                                                                              \
        std::vector<decltype(std::get<0>(*proj.begin()).rule_.param)> data;        \
        data.reserve(proj.size());                                                 \
        std::transform(                                                            \
            proj.begin(), proj.end(), std::back_inserter(data),                    \
            [](const auto &synapse) { return std::get<0>(synapse).rule_.param; }); \
        group.createDataSet(std::string("rule_") + #param, data);                  \
    }                                                                              \
    static_assert(true, "")


#define READ_SYNAPSE_RULE_PARAMETER(target, parameter, h5_group, proj_size, def_value)                         \
    {                                                                                                          \
        const auto values = read_parameter(h5_group, std::string("rule_") + #parameter, proj_size, def_value); \
        for (size_t i = 0; i < target.size(); ++i) std::get<0>(target[i]).rule_.parameter = values[i];         \
    }                                                                                                          \
    static_assert(true, "")


template <>
void add_projection_to_h5<core::Projection<ResourceDeltaSynapse>>(
    HighFive::File &file_h5, const knp::core::Projection<ResourceDeltaSynapse> &projection)
{
    if (!file_h5.exist("edges"))
    {
        throw std::runtime_error("File does not contain the \"edges\" group.");
    }

    std::vector<uint64_t> source_ids, target_ids;
    std::vector<decltype(synapse_traits::synapse_parameters<ResourceDeltaSynapse>::delay_)> delays;
    std::vector<decltype(synapse_traits::synapse_parameters<ResourceDeltaSynapse>::weight_)> weights;
    std::vector<int> out_types;
    source_ids.reserve(projection.size());
    target_ids.reserve(projection.size());
    delays.reserve(projection.size());
    weights.reserve(projection.size());
    out_types.reserve(projection.size());

    for (const auto &v : projection)
    {
        const auto &params = std::get<knp::core::synapse_data>(v);
        source_ids.push_back(std::get<knp::core::source_neuron_id>(v));
        target_ids.push_back(std::get<knp::core::target_neuron_id>(v));
        delays.push_back(params.delay_);
        weights.push_back(params.weight_);
        out_types.push_back(static_cast<int>(std::get<knp::core::synapse_data>(v).output_type_));
    }

    HighFive::Group proj_group = file_h5.createGroup("edges/" + std::string(projection.get_uid()));
    HighFive::DataSet source_node_dataset = proj_group.createDataSet("source_node_id", source_ids);
    source_node_dataset.createAttribute("node_population", std::string(projection.get_presynaptic()));

    HighFive::DataSet target_node_dataset = proj_group.createDataSet("target_node_id", target_ids);
    target_node_dataset.createAttribute("node_population", std::string(projection.get_postsynaptic()));

    proj_group.createDataSet("edge_group_id", std::vector(projection.size(), 0));
    proj_group.createDataSet(
        "edge_type_id", std::vector(projection.size(), get_synapse_type_id<ResourceDeltaSynapse>()));

    std::vector<uint64_t> group_index;
    group_index.reserve(projection.size());
    for (size_t i = 0; i < projection.size(); ++i) group_index.push_back(i);

    proj_group.createDataSet("edge_group_index", group_index);

    HighFive::Group syn_group = proj_group.createGroup("0");
    PUT_SYNAPSE_RULE_TO_DATASET(projection, d_u_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, had_hebbian_update_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, has_contributed_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, synaptic_resource_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, last_spike_step_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, dopamine_plasticity_period_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, w_max_, syn_group);
    PUT_SYNAPSE_RULE_TO_DATASET(projection, w_min_, syn_group);
    proj_group.createAttribute("is_locked", projection.is_locked());

    syn_group.createDataSet("syn_weight", weights);
    syn_group.createDataSet("delay", delays);
    syn_group.createDataSet("output_type_", out_types);
}


template <>
core::Projection<ResourceDeltaSynapse> load_projection(
    const HighFive::Group &edges_group, const std::string &projection_name)
{
    auto projection_group = edges_group.getGroup(projection_name);
    auto group = projection_group.getGroup("0");
    size_t group_size = edges_group.getGroup(projection_name).getDataSet("edge_group_id").getDimensions().at(0);

    // std::vector<core::Projection<ResourceDeltaSynapse>::Synapse> target(group_size);

    const auto weights = read_parameter<decltype(synapse_traits::synapse_parameters<ResourceDeltaSynapse>::weight_)>(
        group, "syn_weight", group_size, synapse_traits::default_values<synapse_traits::DeltaSynapse>::weight_);

    const auto delays = read_parameter<decltype(synapse_traits::synapse_parameters<ResourceDeltaSynapse>::delay_)>(
        group, "delay", group_size, synapse_traits::default_values<synapse_traits::DeltaSynapse>::delay_);

    const auto out_types = read_parameter<int>(
        group, "output_type_", group_size,
        static_cast<int>(synapse_traits::default_values<synapse_traits::DeltaSynapse>::output_type_));

    const auto source_ids = read_parameter<size_t>(projection_group, "source_node_id", group_size, 0);
    const auto target_ids = read_parameter<size_t>(projection_group, "target_node_id", group_size, 0);

    const core::UID uid_from{boost::lexical_cast<boost::uuids::uuid>(
        projection_group.getDataSet("source_node_id").getAttribute("node_population").read<std::string>())};
    const core::UID uid_to{boost::lexical_cast<boost::uuids::uuid>(
        projection_group.getDataSet("target_node_id").getAttribute("node_population").read<std::string>())};
    const core::UID uid_own{boost::lexical_cast<boost::uuids::uuid>(projection_name)};

    std::vector<std::tuple<synapse_traits::synapse_parameters<ResourceDeltaSynapse>, size_t, size_t>> synapses;
    synapses.reserve(group_size);

    for (size_t i = 0; i < weights.size(); ++i)
    {
        synapse_traits::synapse_parameters<ResourceDeltaSynapse> syn;
        syn.weight_ = weights[i];
        syn.delay_ = delays[i];
        syn.output_type_ = static_cast<synapse_traits::OutputType>(out_types[i]);
        synapses.emplace_back(syn, source_ids[i], target_ids[i]);
    }
    static const synapse_traits::synapse_parameters<synapse_traits::SynapticResourceSTDPDeltaSynapse> def_params;

    READ_SYNAPSE_RULE_PARAMETER(synapses, d_u_, group, group_size, def_params.rule_.d_u_);
    READ_SYNAPSE_RULE_PARAMETER(synapses, had_hebbian_update_, group, group_size, def_params.rule_.had_hebbian_update_);
    READ_SYNAPSE_RULE_PARAMETER(synapses, has_contributed_, group, group_size, def_params.rule_.has_contributed_);
    READ_SYNAPSE_RULE_PARAMETER(synapses, synaptic_resource_, group, group_size, def_params.rule_.synaptic_resource_);
    READ_SYNAPSE_RULE_PARAMETER(synapses, last_spike_step_, group, group_size, def_params.rule_.last_spike_step_);
    READ_SYNAPSE_RULE_PARAMETER(
        synapses, dopamine_plasticity_period_, group, group_size, def_params.rule_.dopamine_plasticity_period_);
    READ_SYNAPSE_RULE_PARAMETER(synapses, w_max_, group, group_size, def_params.rule_.w_max_);
    READ_SYNAPSE_RULE_PARAMETER(synapses, w_min_, group, group_size, def_params.rule_.w_min_);


    core::Projection<ResourceDeltaSynapse> proj(
        uid_own, uid_from, uid_to, [&synapses](size_t index) { return synapses[index]; }, synapses.size());
    if (projection_group.hasAttribute("is_locked"))
    {
        if (projection_group.getAttribute("is_locked").read<bool>())
        {
            proj.lock_weights();
        }
        else
        {
            proj.unlock_weights();
        }
    }
    return proj;
}


}  // namespace knp::framework::sonata
