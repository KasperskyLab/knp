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
 * @brief Functor that checks connectivity of a network.
 * 
 * @details The validator inspects all populations and projections in a network and reports an error
 * when a population has no incoming and no outgoing projections, or when a projection has neither
 * a presynaptic nor a postsynaptic population. The result is returned as a 'Report'.
 * 
 * @note The validator does not modify the network, it only reads its structure.
 */
class KNP_DECLSPEC Connectivity
{
public:
    /**
     * @brief Internal error codes used by a validator.
     * 
     * @details These codes are converted to a standard 'std::error_code` via `make_error_code()`. They
     * are also used to select a human-readable message template.
     */
    enum ErrorCode
    {
        /**
         * @brief A population has no connected projections.
         */
        population_not_connected,

        /**
         * @brief A projection has no connected populations.
         */
        projection_not_connected,
    };

    /**
     * @brief Obtain the `std::error_category` associated with this validator.
     * 
     * @details It is used by `make_error_code()` to build a `std::error_code` that can be stored in an
     * `Issue`.
     * 
     * @return reference to the error category object.
     */
    static const std::error_category& error_category() noexcept;

    /**
     * @brief Convert an `ErrorCode` to a standard `std::error_code`.
     * 
     * @param error error code to convert.
     * 
     * @return `std::error_code` belonging to the `Connectivity` error category.
     */
    static std::error_code make_error_code(ErrorCode error) noexcept;

    /**
     * @brief Run the connectivity validation on a given network.
     * 
     * @param network network to validate.
     * 
     * @return `Report` containing all found issues, the report is empty when the network is fully
     * connected.
     */
    Report operator()(const Network& network);
};

}  // namespace knp::framework::network_validation
