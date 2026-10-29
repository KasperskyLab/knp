/**
 * @file synchronization.h
 * @brief Functions to read network back from.
 * @kaspersky_support Artiom N.
 * @date 22.03.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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

#include <knp/core/backend.h>
#include <knp/core/impexp.h>
#include <knp/framework/network.h>


/**
 * @brief Synchronization namespace.
 */
namespace knp::framework::synchronization
{
/**
 * @brief Copy network data from backend.
 * @param backend backend containing a network.
 * @return Network in its current state on backend.
 */
KNP_DECLSPEC Network get_network_copy(const knp::core::Backend &backend);

}  // namespace knp::framework::synchronization
