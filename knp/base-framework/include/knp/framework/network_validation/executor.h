/**
 * @file executor.h
 * @brief Network validators executor.
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

#include <knp/framework/network_validation/validator_types.h>

#include <map>
#include <string>
#include <utility>
#include <vector>


/**
 * @brief Network validation namespace.
 */
namespace knp::framework::network_validation
{

/**
 * @brief Executor of validators.
 */
class KNP_DECLSPEC Executor
{
public:
    /**
     * @brief Type of validator's UID.
     */
    using ValidatorUID = knp::core::UID;

    /**
     * @brief Add validator to run later.
     * @param validator Population validator.
     * @param name Validator's display name.
     * @return Validator UID.
     */
    ValidatorUID add_validator(std::string name, PopulationValidator validator);

    /**
     * @brief Add validator to run later.
     * @note Automatically generates name.
     * @param validator Population validator.
     * @return Validator UID.
     */
    ValidatorUID add_validator(PopulationValidator validator);

    /**
     * @brief Add validator to run later.
     * @param validator Projection validator.
     * @param name Validator's display name.
     * @return Validator UID.
     */
    ValidatorUID add_validator(std::string name, ProjectionValidator validator);

    /**
     * @brief Add validator to run later.
     * @note Automatically generates name.
     * @param validator Projection validator.
     * @return Validator UID.
     */
    ValidatorUID add_validator(ProjectionValidator validator);

    /**
     * @brief Add validator to run later.
     * @param validator Network validator.
     * @param name Validator's display name.
     * @return Validator UID.
     */
    ValidatorUID add_validator(std::string name, NetworkValidator validator);

    /**
     * @brief Add validator to run later.
     * @note Automatically generates name.
     * @param validator Network validator.
     * @return Validator UID.
     */
    ValidatorUID add_validator(NetworkValidator validator);

    /**
     * @brief Validator's report info. Contains name and report.
     */
    struct ValidatorResult
    {
        /**
         * @brief UID.
         */
        // cppcheck-suppress unusedStructMember
        ValidatorUID uid_;

        /**
         * @brief Display name.
         */
        // cppcheck-suppress unusedStructMember
        std::string validator_name_;

        /**
         * @brief Report.
         */
        // cppcheck-suppress unusedStructMember
        Report report_;
    };

    /**
     * @brief Run validators.
     * @param network Network to run validators on.
     * @return Vector of validators reports.
     */
    std::vector<ValidatorResult> run_validators(const Network& network);

private:
    static void log_report(const Report& report);

    // cppcheck-suppress unusedStructMember
    std::map<ValidatorUID, std::pair<std::string, PopulationValidator>> population_validators_;
    // cppcheck-suppress unusedStructMember
    std::map<ValidatorUID, std::pair<std::string, ProjectionValidator>> projection_validators_;
    // cppcheck-suppress unusedStructMember
    std::map<ValidatorUID, std::pair<std::string, NetworkValidator>> network_validators_;
};

}  // namespace knp::framework::network_validation
