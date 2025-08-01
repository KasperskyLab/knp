/**
 * @file construct_network.h
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 03.12.2024
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

#include <knp/framework/network.h>

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>


struct AnnotatedNetwork
{
    knp::framework::Network network_;
    struct Annotation
    {
        std::vector<knp::core::UID> output_uids_;
        std::vector<knp::core::UID> projections_from_raster_;
        std::vector<knp::core::UID> projections_from_classes_;
        std::set<knp::core::UID> inference_population_uids_;
        std::set<knp::core::UID> inference_internal_projection_;

        // For each compound network: a vector of senders and a vector of receivers.
        std::vector<std::pair<std::vector<knp::core::UID>, std::vector<knp::core::UID>>> wta_data_;
        std::map<knp::core::UID, std::string> population_names_;
    } data_;
};


AnnotatedNetwork create_example_network(int num_compound_networks);
