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

#include <knp/framework/network.h>
#include <knp/framework/network_validation/report.h>

#include <string>
#include <system_error>


/**
 * @brief Network validation namespace.
 */
namespace knp::framework::network_validation
{

/**
 * @brief Connectivity validator functor.
 * @note It will test if there are no populations/projections that are not connected to anything.
 */
class Connectivity
{
public:
    /**
     * @brief Internal error codes.
     */
    enum ErrorCode
    {
        population_not_connected,
        projection_not_connected,
    };

    /**
     * @brief Get error category for this class.
     * @return Error category.
     */
    static const std::error_category& error_category() noexcept;

    /**
     * @brief Convert ErrorCode to standard error_code.
     * @param error Error code.
     * @return Converted error code.
     */
    static std::error_code make_error_code(ErrorCode error) noexcept;

    /**
     * @brief Run connectivity validation on specified network.
     * @param network Network for validation.
     * @return Report.
     */
    Report operator()(const Network& network);
};

}  // namespace knp::framework::network_validation
