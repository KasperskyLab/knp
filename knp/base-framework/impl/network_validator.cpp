/**
 * @file network_validator.cpp
 * @brief Network validation interface.
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

#include <knp/framework/network_validator.h>

#include <spdlog/spdlog.h>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{
bool NetworkValidator::run_validators(const Network& network)
{
    SPDLOG_INFO("Starting network validators...");

    bool all_passed = true;
    for (auto& validator : validators_)
    {
        SPDLOG_INFO("[{}]:", validator->get_name());
        bool result = validator->run_validation(network);
        if (result)
        {
            SPDLOG_INFO("PASSED");
        }
        else
        {
            all_passed = false;
        }
    }

    if (all_passed)
    {
        SPDLOG_INFO("All network validators passed");
    }
    else
    {
        SPDLOG_INFO("Not all network validators passed.");
    }

    return all_passed;
}
}  //namespace knp::framework
