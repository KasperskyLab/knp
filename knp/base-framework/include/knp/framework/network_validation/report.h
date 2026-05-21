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
 * @brief Severity of report.
 */
enum class IssueSeverity
{
    info,
    warning,
    error
};


/**
 * @brief Issue from validator.
 */
struct Issue
{
    /**
     * @brief Severity.
     */
    // cppcheck-suppress unusedStructMember
    IssueSeverity severity_;

    /**
     * @brief Message.
     */
    // cppcheck-suppress unusedStructMember
    std::string message_;

    /**
     * @brief Internal code error.
     */
    // cppcheck-suppress unusedStructMember
    std::error_code code_;
};


/**
 * @brief Alias for return type of validator, aka vector of issues.
 */
using Report = std::vector<Issue>;

}  // namespace knp::framework::network_validation
