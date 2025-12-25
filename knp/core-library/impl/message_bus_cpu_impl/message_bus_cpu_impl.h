/**
 * @file message_bus_cpu_impl.h
 * @brief CPU-based message bus implementation header.
 * @kaspersky_support Vartenkov A.
 * @date 18.09.2023
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

#include <knp/core/message_bus.h>

#include <message_bus_impl.h>

#include <spdlog/spdlog.h>

#include <list>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_set>
#include <vector>


/**
 * @brief Namespace for implementations of message bus.
 */
namespace knp::core::messaging::impl
{
class MessageEndpointCPU;

class MessageBusCPUImpl : public MessageBusImpl
{
public:
    MessageBusCPUImpl() { SPDLOG_DEBUG("CPU message bus creating..."); }

    void update() override;
    size_t step() override;
    template <typename MessageType>
    Subscription<MessageType> &subscribe(const UID &receiver, const std::vector<UID> &senders);
    [[nodiscard]] core::MessageEndpoint create_endpoint() override;

private:
    std::vector<knp::core::messaging::MessageVariant> messages_to_route_;

    // This is a list of endpoint data:
    // First vector is a pointer to a set of messages the endpoint is sending.
    // Second vector is a set of messages endpoint is receiving.
    // Third set contains message senders, and is kept updated by endpoints when they add or remove them.
    std::list<std::tuple<
        std::weak_ptr<std::vector<messaging::MessageVariant>>, std::weak_ptr<std::vector<messaging::MessageVariant>>,
        std::weak_ptr<std::unordered_set<knp::core::UID, knp::core::uid_hash>>>>
        endpoint_data_;
    std::mutex mutex_;
};
}  // namespace knp::core::messaging::impl
