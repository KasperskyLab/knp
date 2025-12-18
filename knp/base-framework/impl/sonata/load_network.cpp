/**
 * @file load_network.cpp
 * @brief Network loading source file.
 * @kaspersky_support An. Vartenkov
 * @date 29.02.2024
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

#include "load_network.h"

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/core/uid.h>
#include <knp/framework/network.h>

#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>

#include "highfive.h"
#include "types/type_id_defines.h"


namespace knp::framework::sonata
{
namespace fs = std::filesystem;


std::vector<std::string> get_projection_names(const HighFive::File &file)
{
    auto edges_group = file.getGroup("edges");
    size_t num_projections = edges_group.getNumberObjects();
    std::vector<std::string> result;
    result.reserve(num_projections);
    for (size_t i = 0; i < num_projections; ++i)
    {
        result.push_back(edges_group.getObjectName(i));
    }
    return result;
}


std::vector<core::AllProjectionsVariant> load_projections(const fs::path &proj_h5_file)
{
    if (!fs::is_regular_file(proj_h5_file))
        throw std::runtime_error("Could not open file \"" + proj_h5_file.string() + "\".");
    HighFive::File storage{proj_h5_file.string()};
    auto group = storage.getGroup("edges");
    size_t num_projections = group.getNumberObjects();
    std::vector<core::AllProjectionsVariant> result;

    for (size_t i = 0; i < num_projections; ++i)
    {
        std::string proj_name = group.getObjectName(i);
        int proj_type =
            group.getGroup(proj_name).getDataSet("edge_type_id").read<std::vector<int>>()[0];  // One type only.
        // TODO: Check if type is in type_file.
        if (proj_type == get_synapse_type_id<synapse_traits::DeltaSynapse>())
            result.emplace_back(load_projection<synapse_traits::DeltaSynapse>(group, proj_name));
        else if (proj_type == get_synapse_type_id<synapse_traits::SynapticResourceSTDPDeltaSynapse>())
            result.emplace_back(load_projection<synapse_traits::SynapticResourceSTDPDeltaSynapse>(group, proj_name));
        // TODO: Add other supported types or better use a template.
    }
    return result;
}


std::vector<core::AllPopulationsVariant> load_populations(const fs::path &pop_h5_file)
{
    if (!fs::is_regular_file(pop_h5_file))
    {
        throw std::runtime_error("Could not open file \"" + pop_h5_file.string() + "\".");
    }

    const HighFive::File storage{pop_h5_file.string()};
    auto group = storage.getGroup("nodes");
    const size_t num_populations = group.getNumberObjects();
    std::vector<core::AllPopulationsVariant> result;

    for (size_t i = 0; i < num_populations; ++i)
    {
        const std::string proj_name = group.getObjectName(i);
        const int neuron_type =
            group.getGroup(proj_name).getDataSet("node_type_id").read<std::vector<int>>()[0];  // One type only.
        // Check if type is in type_file.
        if (neuron_type == get_neuron_type_id<neuron_traits::BLIFATNeuron>())
            result.emplace_back(load_population<neuron_traits::BLIFATNeuron>(group, proj_name));
        else if (neuron_type == get_neuron_type_id<neuron_traits::SynapticResourceSTDPBLIFATNeuron>())
            result.emplace_back(load_population<neuron_traits::SynapticResourceSTDPBLIFATNeuron>(group, proj_name));
        else if (neuron_type == get_neuron_type_id<neuron_traits::AltAILIF>())
            result.emplace_back(load_population<neuron_traits::AltAILIF>(group, proj_name));
        // TODO: Add other supported types or better use a template.
    }
    return result;
}


struct NetworkConfig
{
    const fs::path config_path;
    const fs::path edges_storage;
    const fs::path nodes_storage;
    const fs::path edges_types;
    const fs::path nodes_types;
};


NetworkConfig read_config_file(const fs::path &config_path)
{
    const fs::path network_dir = config_path.parent_path();
    const fs::path edges_storage = network_dir / "projections.h5";
    const fs::path nodes_storage = network_dir / "populations.h5";
    const fs::path edges_types = network_dir / "synapses.csv";
    const fs::path nodes_types = network_dir / "neurons.csv";
    return NetworkConfig{config_path, edges_storage, nodes_storage, edges_types, nodes_types};
    // TODO: Actually read config.
}


core::UID get_network_uid(const fs::path &nodes_path)
{
    const HighFive::File h5_file(nodes_path.string());
    if (h5_file.hasAttribute("network_uid"))
    {
        auto uid_str = h5_file.getAttribute("network_uid").read<std::string>();
        return core::UID{boost::lexical_cast<boost::uuids::uuid>(uid_str)};
    }
    return core::UID{};
}


KNP_DECLSPEC Network load_network(const fs::path &config_path)
{
    // TODO: Get this value from config file at config_path.
    const std::string config_path_suffix = "network/network_config.json";
    // Open and read config files.
    auto config = read_config_file(config_path / config_path_suffix);
    Network network{get_network_uid(config.nodes_storage)};
    auto populations = load_populations(config.nodes_storage);
    auto projections = load_projections(config.edges_storage);
    for (auto &pop : populations)
    {
        std::visit([&network](auto &population) { network.add_population(population); }, pop);
    }

    for (auto &proj : projections)
    {
        std::visit([&network](auto &projection) { network.add_projection(projection); }, proj);
    }

    return network;
}

}  // namespace knp::framework::sonata
