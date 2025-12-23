/**
 * @file load_network.h
 * @brief Loading network to SONATA format file.
 * @kaspersky_support An. Vartenkov
 * @date 15.04.2024
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

#pragma once

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "highfive.h"


namespace knp::framework::sonata
{
namespace fs = std::filesystem;

template <class Neuron>
core::Population<Neuron> load_population(const HighFive::Group &nodes_group, const std::string &population_name);

template <class Synapse>
core::Projection<Synapse> load_projection(const HighFive::Group &edges_group, const std::string &projection_name);

}  // namespace knp::framework::sonata


#define LOAD_NEURONS_PARAMETER(target, neuron_type, parameter, h5_group, pop_size)                                 \
    do                                                                                                             \
    {                                                                                                              \
        const auto values =                                                                                        \
            read_parameter(h5_group, #parameter, pop_size, neuron_traits::default_values<neuron_type>::parameter); \
        for (size_t i = 0; i < std::min(values.size(), target.size()); ++i) target[i].parameter = values[i];       \
    } while (false)


#define LOAD_SYNAPSE_PARAMETER(target, synapse_type, parameter, h5_group, proj_size)                                  \
    do                                                                                                                \
    {                                                                                                                 \
        const auto values =                                                                                           \
            read_parameter(h5_group, #parameter, proj_size, synapse_traits::default_values<synapse_type>::parameter); \
        for (size_t i = 0; i < std::min(values.size(), target.size()); ++i) target[i].parameter = values[i];          \
    } while (false)
