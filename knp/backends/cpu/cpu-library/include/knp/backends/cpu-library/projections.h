/**
 * @file projections.h
 * @kaspersky_support Postnikov D.
 * @date 10.12.2025
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

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <spdlog/spdlog.h>

#include "temp_impl/projections/interface.h"


namespace knp::backends::cpu::projections
{

/**
 * @brief Calculate projection.
 * @param projection Projection.
 * @param endpoint Projection endpoint.
 * @param future_messages Future messages queue.
 * @param step_n Step number.
 */
template <typename Synapse>
void calculate_projection(
    knp::core::Projection<Synapse> &projection, knp::core::MessageEndpoint &endpoint, MessageQueue &future_messages,
    size_t step_n)
{
    SPDLOG_DEBUG("Calculating delta synapse projection...");

    auto messages = endpoint.unload_messages<core::messaging::SpikeMessage>(projection.get_uid());
    auto out_iter = calculate_projection_interface(projection, messages, future_messages, step_n);
    if (out_iter != future_messages.end())
    {
        SPDLOG_TRACE("Projection is sending an impact message.");
        // Send a message and remove it from the queue.
        endpoint.send_message(out_iter->second);
        future_messages.erase(out_iter);
    }
}

}  //namespace knp::backends::cpu::projections
