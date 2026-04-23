/**
 * @file projections.h
 * @brief Interface for working with projections.
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

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <spdlog/spdlog.h>

#include <string>
#include <unordered_map>

#include "impl/projections/projection_dispatcher.h"


/**
 * @brief Namespace for CPU backend projection functions.
 */
namespace knp::backends::cpu::projections
{

/**
 * @brief Calculate a synapse projection for the given simulation step.
 *
 * @tparam Synapse type of a synapse that possesses Delta‑like parameters.
 *
 * @param projection projection to calculate.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param future_messages queue that stores messages to be sent in future steps.
 * @param step_n current simulation step number.
 *
 * @details First, the function unloads all spike messages addressed to the projection from
 * the message endpoint. Then the function processes the unloaded spike messages together with 
 * any pending `future_messages` and returns an iterator to the first impact message that should be
 *  sent immediately. If such a message exists, the function sends it via the message endpoint
 *  and removes it from `future_messages`.
 */
template <typename Synapse>
void calculate_projection(
    knp::core::Projection<Synapse> &projection, knp::core::MessageEndpoint &endpoint, MessageQueue &future_messages,
    size_t step_n)
{
    SPDLOG_DEBUG("Calculating synapse projection at step {}.", step_n);

    auto messages = endpoint.unload_messages<core::messaging::SpikeMessage>(projection.get_uid());

#if (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE)
    for (const auto &message : messages)
    {
        SPDLOG_TRACE(
            "Spike from {} with spiked neurons: {}", std::string(message.header_.sender_uid_),
            fmt::join(message.neuron_indexes_, ", "));
    }
#endif

    auto out_iter = impl::calculate_projection_dispatch(projection, messages, future_messages, step_n);
    if (out_iter != future_messages.end())
    {
        SPDLOG_TRACE("Projection is sending an impact message.");
        // Send a message and remove it from the queue.
        endpoint.send_message(out_iter->second);
        future_messages.erase(out_iter);
    }
}


/**
 * @brief Process a part of projection synapses in a multi-threaded way.
 *
 * @todo Get rid of this function. This function exists only to keep the multi-threaded backend functional.
 *
 * @tparam Synapse type of synapses stored in the projection.
 *
 * @param projection projection that will receive the processed messages.
 * @param message_in_data processed spike data for the projection.
 * @param future_messages queue of future messages.
 * @param step_n current simulation step.
 * @param part_start index of the starting synapse.
 * @param part_size number of synapses to process.
 * @param mutex mutex used for synchronization.
 *
 */
template <class Synapse>
void calculate_projection_multithreaded(
    knp::core::Projection<Synapse> &projection, const std::unordered_map<knp::core::Step, size_t> &message_in_data,
    MessageQueue &future_messages, uint64_t step_n, size_t part_start, size_t part_size, std::mutex &mutex)
{
    impl::calculate_projection_multithreaded_dispatch(
        projection, message_in_data, future_messages, step_n, part_start, part_size, mutex);
}

}  //namespace knp::backends::cpu::projections
