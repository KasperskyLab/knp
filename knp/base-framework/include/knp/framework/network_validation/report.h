/**
 * @file report.h
 * @brief Report type.
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

#include <string>
#include <system_error>
#include <vector>


/**
 * @brief Network validation namespace.
 */
namespace knp::framework::network_validation
{

/**
 * @brief Severity levels for validation issues.
 * 
 * @details The enumeration is used by validators to classify each reported problem.
 */
enum class IssueSeverity
{
    /**
     * @brief Informational message.
     */
    info,

    /**
     * @brief Potential problem that does not stop execution.
     */
    warning,

    /**
     * @brief Serious problem that usually aborts the validation run.
     */
    error
};


/**
 * @brief Single issue reported by validator.
 * 
 * @details An 'Issue' aggregates a severity, a human-readable message, and 'std::error_code'
 * that identifies the underlying error.
 */
struct Issue
{
    /**
     * @brief Severity of the issue.
     */
    // cppcheck-suppress unusedStructMember
    IssueSeverity severity_;

    /**
     * @brief Human-readable description of the issue.
     */
    // cppcheck-suppress unusedStructMember
    std::string message_;

    /**
     * @brief Machine-readable error code associated with the issue.
     */
    // cppcheck-suppress unusedStructMember
    std::error_code code_;
};


/**
 * @brief Alias for the validator return type.
 * 
 * @details A validator returns a 'Report', which is simply a 'std::vector<Issue>'. An empty
 * report indicates that the validator found no problems.
 */
using Report = std::vector<Issue>;

}  // namespace knp::framework::network_validation
