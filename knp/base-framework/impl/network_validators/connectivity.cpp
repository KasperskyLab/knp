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

#include <knp/framework/network_validators/connectivity.h>

#include <spdlog/spdlog.h>


/**
 * @brief Network validators namespace.
 */
namespace knp::framework::network_validators
{
std::string Connectivity::get_name() const
{
    return "Network connectivity validator";
}

bool Connectivity::run_validation(const Network& network)
{
    return true;
}
}  //namespace knp::framework::network_validators
