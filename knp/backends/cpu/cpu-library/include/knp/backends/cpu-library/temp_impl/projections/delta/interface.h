/**
 * @file interface.h
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

// This include does not work at the end of the day. But its here so code analyzer will work properly.
#include <knp/backends/cpu-library/temp_impl/projections/interface.h>

#include <vector>

#include "impl.h"

namespace knp::backends::cpu::projections
{

/**
 * @brief Calculate projection.
 * @param projection Projection.
 * @param messages Incoming messages.
 * @param future_messages Messages queue for future.
 * @param step_n Step number.
 * @return Message that should be sent from queue.
 */
template <>
inline MessageQueue::const_iterator calculate_projection_interface<delta::DeltaSynapse>(
    knp::core::Projection<delta::DeltaSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    MessageQueue &future_messages, size_t step_n)
{
    return delta::calculate_projection_impl(projection, messages, future_messages, step_n);
}

/**
 * @brief Calculate projection.
 * @param projection Projection.
 * @param messages Incoming messages.
 * @param future_messages Messages queue for future.
 * @param step_n Step number.
 * @return Message that should be sent from queue.
 */
template <>
inline MessageQueue::const_iterator calculate_projection_interface<delta::STDPDeltaSynapse>(
    knp::core::Projection<delta::STDPDeltaSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    MessageQueue &future_messages, size_t step_n)
{
    return delta::calculate_projection_impl(projection, messages, future_messages, step_n);
}

}  //namespace knp::backends::cpu::projections
