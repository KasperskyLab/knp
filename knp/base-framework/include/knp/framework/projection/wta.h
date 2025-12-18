/**
 * @file wta.h
 * @brief Functions for Winner Takes All.
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

#pragma once

#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>

#include <utility>
#include <vector>


/**
 * @brief Projection namespace.
 */
namespace knp::framework::projection
{

/**
 * @brief Add Winner-Takes-All (WTA) handlers to a network.
 * @details The WTA handlers are added for each compound network specified in @p wta_data which contains pairs of senders 
 * and receivers. @p borders specifies the borders for the WTA behavior, and @p winners_amount specifies the number 
 * of winners to select.
 * @param executor model executor to which the WTA handlers will be added.
 * @param winners_amount number of winners to select for each WTA group.
 * @param borders borders for the WTA behavior, which determine the scope of the WTA competition.
 * @param wta_data vector of pairs, where each pair contains a vector of senders and a vector of receivers for a compound network.
 * @return vector of UIDs for the added WTA handlers.
 */
KNP_DECLSPEC std::vector<knp::core::UID> add_wta_handlers(
    knp::framework::ModelExecutor& executor, size_t winners_amount, const std::vector<size_t>& borders,
    const std::vector<std::pair<std::vector<knp::core::UID>, std::vector<knp::core::UID>>>& wta_data);

}  // namespace knp::framework::projection
