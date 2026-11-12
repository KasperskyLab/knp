/**
 * @file type_id_defines.h
 * @brief Basic type definitions and templates for saving and loading.
 * @kaspersky_support A. Vartenkov
 * @date 22.03.2024
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

#pragma once

#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "../csv_content.h"


namespace knp::framework::sonata
{
constexpr int BASE_TYPE_ID = 100;

template <class Synapse>
constexpr int get_synapse_type_id()
{
    return boost::mp11::mp_find<synapse_traits::AllSynapses, Synapse>() + BASE_TYPE_ID;
}


template <class Synapse>
std::string get_synapse_type_name();


template <class Neuron>
constexpr int get_neuron_type_id()
{
    return boost::mp11::mp_find<core::AllCells, Neuron>() + BASE_TYPE_ID;
}


template <class Synapse>
std::string get_neuron_type_name();


template <class Neuron>
void add_neuron_type_to_csv(const fs::path &csv_path)
{
    std::vector<std::string> type_row = {
        std::to_string(get_neuron_type_id<Neuron>()), "point_neuron", "", get_neuron_type_name<Neuron>()};

    if (!fs::is_regular_file(csv_path))
    {
        // No file, need to create one.
        CsvContent csv_file;
        csv_file.set_header(node_file_header);
        csv_file.add_row(type_row);
        save_csv_content(csv_file, csv_path);
    }

    // File already exists, load it.
    CsvContent csv_file = load_csv_content(csv_path);
    // Check header correctness.
    auto file_header = csv_file.get_header();
    for (const auto &column_name : node_file_header)
    {
        if (std::find(file_header.begin(), file_header.end(), column_name) ==
            file_header.end())  // cppcheck-suppress useStlAlgorithm
            throw std::runtime_error(
                "Could not find column \"" + column_name + "\" in file \"" + csv_path.string() + "\".");
    }
    // Header is okay, check if type exists already.
    size_t height = csv_file.get_rc_size().first - 1;
    for (size_t row_id = 0; row_id < height; ++row_id)
    {
        if (csv_file.get_value<int>(row_id, "node_type_id") == get_neuron_type_id<Neuron>())
        {
            return;  // Type exists, nothing to add.
        }
    }
    csv_file.add_row(type_row);
    save_csv_content(csv_file, csv_path);
}


template <class Synapse>
void add_synapse_type_to_csv(const fs::path &csv_path)
{
    if (fs::is_regular_file(csv_path))
    {
        // File already exists, load it.
        CsvContent csv_file = load_csv_content(csv_path);
        // Check header correctness.
        auto file_header = csv_file.get_header();
        for (const auto &column_name : edge_file_header)
        {
            if (std::find(file_header.begin(), file_header.end(), column_name) ==
                file_header.end())  // cppcheck-suppress useStlAlgorithm
                throw std::runtime_error(
                    "Could not find column \"" + column_name + "\" in file \"" + csv_path.string() + "\".");
        }
        // Header is okay, check if type exists already.
        size_t height = csv_file.get_rc_size().first - 1;
        for (size_t row_id = 0; row_id < height; ++row_id)
        {
            int type_id = csv_file.get_value<int>(row_id, "edge_type_id");
            if (type_id == get_synapse_type_id<Synapse>())
            {
                return;  // Type exists, nothing to add. TODO: Update.
            }
        }
        std::vector<std::string> type_row{
            std::to_string(get_synapse_type_id<Synapse>()), "", get_synapse_type_name<Synapse>()};
        csv_file.add_row(type_row);
        save_csv_content(csv_file, csv_path);
        return;
    }
    // File doesn't exist.
    CsvContent csv_file;
    csv_file.set_header(edge_file_header);
    std::vector<std::string> type_row{
        std::to_string(get_synapse_type_id<Synapse>()), "", get_synapse_type_name<Synapse>()};
    csv_file.add_row(type_row);
    save_csv_content(csv_file, csv_path);
}

}  // namespace knp::framework::sonata
