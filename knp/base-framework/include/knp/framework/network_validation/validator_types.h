/**
 * @file validator_types.h
 * @brief Validator types.
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
#include <knp/framework/network_validation/report.h>

/**
 * @brief Network validation namespace.
 */
namespace knp::framework::network_validation
{

/**
 * @brief Validator for populations.
 */
using PopulationValidator = std::function<Report(const Network::AllPopulationVariants& population)>;


/**
 * @brief Validator for projections.
 */
using ProjectionValidator = std::function<Report(const Network::AllProjectionVariants& projection)>;


/**
 * @brief Validator for network.
 */
using NetworkValidator = std::function<Report(const Network& network)>;

}  // namespace knp::framework::network_validation
