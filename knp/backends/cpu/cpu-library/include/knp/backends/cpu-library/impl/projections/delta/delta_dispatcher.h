/**
 * @file delta_dispatcher.h
 * @brief Specification of projection interface for delta synapse projection.
 * @kaspersky_support Postnikov D.
 * @date 10.12.2025
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

#include <unordered_map>
#include <vector>

#include "delta_impl.h"


namespace knp::backends::cpu::projections::impl
{

/**
 * @brief Calculate projection.
 * @param projection Projection.
 * @param messages Incoming messages.
 * @param future_messages Messages queue for future.
 * @param step_n Step number.
 * @return Message that should be sent from queue.
 */
inline MessageQueue::const_iterator calculate_projection_dispatch(
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
inline MessageQueue::const_iterator calculate_projection_dispatch(
    knp::core::Projection<delta::STDPDeltaSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
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
inline MessageQueue::const_iterator calculate_projection_dispatch(
    knp::core::Projection<delta::AdditiveSTDPDeltaSynapse> &projection,
    std::vector<core::messaging::SpikeMessage> &messages, MessageQueue &future_messages, size_t step_n)
{
    return delta::calculate_projection_impl(projection, messages, future_messages, step_n);
}


/**
 * @brief Process a part of projection synapses in multithreaded way.
 * @param projection projection to receive the message.
 * @param message_in_data processed spike data for the projection.
 * @param future_messages queue of future messages.
 * @param step_n current step.
 * @param part_start index of the starting synapse.
 * @param part_size number of synapses to process.
 * @param mutex mutex.
 */
inline void calculate_projection_multithreaded_dispatch(
    knp::core::Projection<delta::DeltaSynapse> &projection,
    const std::unordered_map<knp::core::Step, size_t> &message_in_data, MessageQueue &future_messages, uint64_t step_n,
    size_t part_start, size_t part_size, std::mutex &mutex)
{
    delta::calculate_projection_multithreaded_impl(
        projection, message_in_data, future_messages, step_n, part_start, part_size, mutex);
}


/**
 * @brief Process a part of projection synapses in multithreaded way.
 * @param projection projection to receive the message.
 * @param message_in_data processed spike data for the projection.
 * @param future_messages queue of future messages.
 * @param step_n current step.
 * @param part_start index of the starting synapse.
 * @param part_size number of synapses to process.
 * @param mutex mutex.
 */
inline void calculate_projection_multithreaded_dispatch(
    knp::core::Projection<delta::STDPDeltaSynapse> &projection,
    const std::unordered_map<knp::core::Step, size_t> &message_in_data, MessageQueue &future_messages, uint64_t step_n,
    size_t part_start, size_t part_size, std::mutex &mutex)
{
    delta::calculate_projection_multithreaded_impl(
        projection, message_in_data, future_messages, step_n, part_start, part_size, mutex);
}

}  //namespace knp::backends::cpu::projections::impl
