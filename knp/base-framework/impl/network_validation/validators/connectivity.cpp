/**
 * @file connectivity.cpp
 * @brief Validator for checking if all projections/populations connected with something.
 * @kaspersky_support David P.
 * @date 03.04.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <knp/framework/network_validation/types.h>
#include <knp/framework/network_validation/validators/connectivity.h>


namespace knp::framework::network_validation
{

std::vector<Report> Connectivity::operator()(const Network& network)
{
    std::vector<Report> report;

    /*
     * We will store a pair of bools for each population, first bool will be true if there is a projection coming out of
     * this population, and second bool will be true if there is a projection coming in this population.
     */
    std::map<knp::core::UID, std::pair<bool, bool>> populations_info;

    /*
     * Same idea here but for projections.
     */
    std::map<knp::core::UID, std::pair<bool, bool>> projections_info;

    const auto& populations = network.get_populations();
    const auto& projections = network.get_projections();

    // Fill empty populations info.
    for (const auto& population_variant : populations)
    {
        std::visit(
            [&populations_info](auto&& population) {
                populations_info[population.get_uid()] = {false, false};
            },
            population_variant);
    }

    // Fill projections and populations info.
    for (const auto& projection_variant : projections)
    {
        std::visit(
            [&projections_info, &populations_info, &report](auto&& projection)
            {
                bool presynaptic_pop_not_empty = static_cast<bool>(projection.get_presynaptic());
                bool postsynaptic_pop_not_empty = static_cast<bool>(projection.get_postsynaptic());
                projections_info[projection.get_uid()] = {presynaptic_pop_not_empty, postsynaptic_pop_not_empty};
                if (presynaptic_pop_not_empty)
                {
                    populations_info[projection.get_presynaptic()].second = true;
                }

                if (postsynaptic_pop_not_empty)
                {
                    populations_info[projection.get_postsynaptic()].first = true;
                }

                if (!presynaptic_pop_not_empty && !postsynaptic_pop_not_empty)
                {
                    report.push_back(
                        {ReportSeverity::error, "Projection " + std::string(projection.get_uid()) +
                                                    " does not have any connected populations."});
                }
            },
            projection_variant);
    }

    //Check if all populations are connected.
    for (const auto& population_info : populations_info)
    {
        if (!population_info.second.first && !population_info.second.second)
        {
            report.push_back(
                {ReportSeverity::error, "Population " + std::string(population_info.first) +
                                            " does not have any projections connected to it."});
        }
    }

    return report;
}

}  // namespace knp::framework::network_validation
