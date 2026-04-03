/**
 * @file network_validator.h
 * @brief Network validation interface.
 * @kaspersky_support David P.
 * @date 02.04.2026
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
#include <knp/framework/network_validators/connectivity.h>

#include <memory>
#include <utility>
#include <vector>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{
class KNP_DECLSPEC NetworkValidator final
{
public:
    template <typename ValidatorType>
    void add_validator(ValidatorType&& validator)
    {
        using DecayedT = std::decay_t<ValidatorType>;
        static_assert(
            std::is_base_of_v<network_validators::Base, DecayedT>,
            "ValidatorType is not derived from validator base class.");
        validators_.push_back(std::make_unique<DecayedT>(std::forward<ValidatorType>(validator)));
    }

    bool run_validators(const Network& network);

private:
    std::vector<std::unique_ptr<network_validators::Base>> validators_;
};
}  //namespace knp::framework
