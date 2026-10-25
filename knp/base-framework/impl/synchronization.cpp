/**
 * @file synchronization.cpp
 * @brief Getting network from backend.
 * @kaspersky_support Vartenkov Andrey.
 * @date 13.05.2024
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

#include <knp/framework/network.h>

namespace knp::framework::synchronization
{
KNP_DECLSPEC Network get_network_copy(const knp::core::Backend &backend)
{
    auto data_ranges = backend.get_network_data();
    knp::framework::Network res_network;
    for (auto &iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        auto population = *iter;
        res_network.add_population(std::move(population));
    }
    for (auto &iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        auto projection = *iter;
        res_network.add_projection(std::move(projection));
    }
    return res_network;
}
}  // namespace knp::framework::synchronization
