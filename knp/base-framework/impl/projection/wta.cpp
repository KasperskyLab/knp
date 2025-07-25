/**
 * @file wta.cpp
 * @brief Functions for Winner Takes All implementation.
 * @kaspersky_support D. Postnikov
 * @date 03.07.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include <knp/framework/monitoring/observer.h>
#include <knp/framework/projection/wta.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/synapse-traits/all_traits.h>

namespace knp::framework::projection
{


std::vector<knp::core::UID> add_wta_handlers(
    knp::framework::ModelExecutor& executor, size_t winners_amount, std::vector<size_t> const& borders,
    std::vector<std::pair<std::vector<knp::core::UID>, std::vector<knp::core::UID>>> const& wta_data)
{
    std::vector<knp::core::UID> result;

    // Generating seed for WTA randomness
    std::mt19937 rand_gen(std::random_device{}());
    std::uniform_int_distribution<int> distr(-std::numeric_limits<int>::max(), std::numeric_limits<int>::max());

    for (const auto& senders_receivers : wta_data)
    {
        knp::core::UID handler_uid;
        executor.add_spike_message_handler(
            knp::framework::modifier::KWtaPerGroup{borders, winners_amount, distr(rand_gen)}, senders_receivers.first,
            senders_receivers.second, handler_uid);
        result.push_back(handler_uid);
    }
    return result;
}

}  // namespace knp::framework::projection
