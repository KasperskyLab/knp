/**
 * @file messaging.h
 * @brief Common header for messaging.
 * @kaspersky_support Artiom N.
 * @date 02.03.2023
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

#include <knp/core/messaging/spike_message.h>
#include <knp/core/messaging/synaptic_impact_message.h>

#include <boost/mp11.hpp>


/**
 * @brief Messaging namespace.
 */
namespace knp::core::messaging
{

/**
 * @brief List of all message types.
 */
#define ALL_MESSAGES SpikeMessage, SynapticImpactMessage

/**
 * @brief List of `boost::mp11` types. You can use the list to manage your message types.
 */
using AllMessages = boost::mp11::mp_list<ALL_MESSAGES>;

}  // namespace knp::core::messaging
