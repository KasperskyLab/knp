/**
 * @file types.h
 * @brief Validators helper types.
 * @kaspersky_support David P.
 * @date 08.04.2026
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

#pragma once

#include <knp/framework/network.h>

#include <string>
#include <vector>


/**
 * @brief Network validation namespace.
 */
namespace knp::framework::network_validation
{

/**
 * @brief Severity of report.
 */
enum ReportSeverity
{
    info,
    warning,
    error
};


/**
 * @brief Report of validator.
 */
struct Report
{
    /**
     * @brief Severity of report.
     */
    // cppcheck-suppress unusedStructMember
    ReportSeverity severity_;


    /**
     * @brief Message of report.
     */
    // cppcheck-suppress unusedStructMember
    std::string message_;
};


/**
 * @brief Validator for populations.
 */
using PopulationValidator = std::function<std::vector<Report>(const Network::AllPopulationVariants& population)>;


/**
 * @brief Validator for projections.
 */
using ProjectionValidator = std::function<std::vector<Report>(const Network::AllProjectionVariants& projection)>;


/**
 * @brief Validator for network.
 */
using NetworkValidator = std::function<std::vector<Report>(const Network& network)>;

}  // namespace knp::framework::network_validation
