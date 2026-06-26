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
 *
 * @details An 'Executor' stores validator callables together with human-readable name and
 * a generated UID. Later the stored validators can be run against a 'Network' instance
 * via 'run_validators()'. The class does not own the network, it only receives a reference
 * when validation is performed.
 */
class KNP_DECLSPEC Executor
{
public:
    /**
     * @brief Add a population validator with an explicit display name.
     *
     * @param validator population validator.
     * @param name validator's display name.
     *
     * @return UID of the added validator.
     */
    knp::core::UID add_validator(std::string name, PopulationValidator validator);

    /**
     * @brief Add a population validator and generate a default name automatically.
     *
     * @note The generated name has the form "Population validator #index", where `index`
     * is the current size of the population-validator map.
     *
     * @param validator population validator.
     *
     * @return UID of the added validator.
     */
    knp::core::UID add_validator(PopulationValidator validator);

    /**
     * @brief Add a projection validator with an explicit display name.
     *
     * @param validator projection validator.
     * @param name validator's display name.
     *
     * @return UID of the added validator.
     */
    knp::core::UID add_validator(std::string name, ProjectionValidator validator);

    /**
     * @brief Add a projection validator and generate a default name automatically.
     *
     * @note The generated name has the form "Projection validator #index", where `index`
     * is the current size of the projection-validator map.
     *
     * @param validator projection validator.
     *
     * @return UID of the added projection.
     */
    knp::core::UID add_validator(ProjectionValidator validator);

    /**
     * @brief Add a network validator with an explicit display name.
     *
     * @param validator network validator.
     * @param name validator's display name.
     *
     * @return UID of the added validator.
     */
    knp::core::UID add_validator(std::string name, NetworkValidator validator);

    /**
     * @brief Add a network validator and generate a default name automatically.
     *
     * @note The generated name has the form "Network validator #index", where `index`
     * is the current size of the network-validator map.
     *
     * @param validator network validator.
     *
     * @return UID of the added validator.
     */
    knp::core::UID add_validator(NetworkValidator validator);

    /**
     * @brief Description of a validator's execution result.
     *
     * @details The structure groups the validator UID, its display name, and the report
     * produced after execution. It is used as the element type of the vector returned
     * by 'run_validators()'.
     */
    struct ValidatorResult
    {
        /**
         * @brief Validator UID.
         */
        // cppcheck-suppress unusedStructMember
        knp::core::UID uid_;

        /**
         * @brief Validator display name.
         */
        // cppcheck-suppress unusedStructMember
        std::string validator_name_;

        /**
         * @brief Validator execution report.
         */
        // cppcheck-suppress unusedStructMember
        Report report_;
    };

    /**
     * @brief Run all stored validators on the supplied network.
     *
     * @details The method iterates over all stored validators. For each validator it logs the
     * start, invokes the validator callable, logs individual issues, and stores a 'ValidatorResult'
     * in the returned vector.
     *
     * @param network network to validate.
     *
     * @return vector of validator results, one entry per executed validator.
     */
    std::vector<ValidatorResult> run_validators(const Network& network);

private:
    static void log_report(const Report& report);

    // cppcheck-suppress unusedStructMember
    std::map<knp::core::UID, std::pair<std::string, PopulationValidator>> population_validators_;
    // cppcheck-suppress unusedStructMember
    std::map<knp::core::UID, std::pair<std::string, ProjectionValidator>> projection_validators_;
    // cppcheck-suppress unusedStructMember
    std::map<knp::core::UID, std::pair<std::string, NetworkValidator>> network_validators_;
};

}  // namespace knp::framework::network_validation
