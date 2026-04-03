/**
 * @file connectivity.h
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

#pragma once

#include <knp/framework/network_validators/base.h>

#include <string>


/**
 * @brief Network validators namespace.
 */
namespace knp::framework::network_validators
{

/**
 * @brief Validator of network connectivity.
 * @details Checks if all populations and projections are connected.
 */
class KNP_DECLSPEC Connectivity final : public Base
{
public:
    /**
     * @brief Get name of validator for logs.
     * @return Validator name.
     */
    [[nodiscard]] std::string get_name() const override;


    /**
     * @brief Run connectivity validation on specified network.
     * @param network Network for validation.
     * @return Result of validation.
     */
    [[nodiscard]] bool run_validation(const Network& network) override;
};
}  //namespace knp::framework::network_validators
