/**
 * @file runner.h
 * @brief Network validators runner.
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

#include <knp/framework/network_validation/types.h>

#include <string>
#include <utility>
#include <vector>


/**
 * @brief Network validation namespace.
 */
namespace knp::framework::network_validation
{

/**
 * @brief Runner of validators.
 */
class KNP_DECLSPEC Runner
{
public:
    /**
     * @brief Add validator to run later.
     * @param name Validator's display name.
     * @param validator Population validator.
     */
    void add_validator(std::string_view name, PopulationValidator validator);


    /**
     * @brief Add validator to run later.
     * @param name Validator's display name.
     * @param validator Projection validator.
     */
    void add_validator(std::string_view name, ProjectionValidator validator);


    /**
     * @brief Add validator to run later.
     * @param name Validator's display name.
     * @param validator Network validator.
     */
    void add_validator(std::string_view name, NetworkValidator validator);


    /**
     * @brief Validator's report info. Contains name and report.
     */
    struct ValidatorReport
    {
        /**
         * @brief Validator's display name.
         */
        // cppcheck-suppress unusedStructMember
        std::string validator_name_;


        /**
         * @brief Validator's report.
         */
        // cppcheck-suppress unusedStructMember
        std::vector<Report> report_;
    };


    /**
     * @brief Run validators.
     * @param network Network to run validators on.
     * @return Vector of validators reports.
     */
    std::vector<ValidatorReport> run_validators(const Network& network);

private:
    static void log_reports(const std::vector<Report>& reports);

    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<std::string, PopulationValidator>> population_validators_;
    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<std::string, ProjectionValidator>> projection_validators_;
    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<std::string, NetworkValidator>> network_validators_;
};

}  // namespace knp::framework::network_validation
