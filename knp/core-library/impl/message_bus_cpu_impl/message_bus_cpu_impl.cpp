/**
 * @file message_bus_cpu_impl.cpp
 * @brief CPU-based message bus implementation.
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

#include <message_bus_cpu_impl/message_bus_cpu_impl.h>
#include <message_bus_cpu_impl/message_endpoint_cpu_impl.h>

#include <utility>


/**
 * @brief Namespace for implementations of message bus.
 */
namespace knp::core::messaging::impl
{

class MessageEndpointCPU : public MessageEndpoint
{
public:
    explicit MessageEndpointCPU(std::shared_ptr<MessageEndpointCPUImpl> &&ptr) { impl_ = std::move(ptr); }
};


void MessageBusCPUImpl::update()
{
    std::lock_guard lock(mutex_);
    // This function is called before routing messages.
    auto iter = endpoint_data_.begin();
    while (iter != endpoint_data_.end())
    {
        auto send_container_ptr = std::get<0>(*iter).lock();
        // Clear up all pointers to expired endpoints.
        if (!send_container_ptr)
        {
            endpoint_data_.erase(iter++);
            continue;
        }

        // Read all sent messages to an internal buffer.
        messages_to_route_.insert(
            messages_to_route_.end(), std::make_move_iterator(send_container_ptr->begin()),
            std::make_move_iterator(send_container_ptr->end()));
        send_container_ptr->clear();
        ++iter;
    }
}


size_t MessageBusCPUImpl::step()
{
    const std::lock_guard lock(mutex_);
    if (messages_to_route_.empty()) return 0;  // No more messages left for endpoints to receive.
    // Sending a message to every endpoint.
    using MessageType = decltype(messages_to_route_)::value_type;
    MessageType message;
    std::swap(message, messages_to_route_.back());
    // Remove message from container.
    messages_to_route_.pop_back();
    const knp::core::UID sender_uid = std::visit([](const auto &msg) { return msg.header_.sender_uid_; }, message);
    for (auto endpoint_data_containers : endpoint_data_)
    {
        auto recv_ptr = std::get<1>(endpoint_data_containers).lock();
        auto allowed_senders_ptr = std::get<2>(endpoint_data_containers).lock();
        // Skip all endpoints deleted after previous update(). They will be deleted at the next update().
        if ((!recv_ptr) || (!allowed_senders_ptr)) continue;
        if (allowed_senders_ptr->find(sender_uid) != allowed_senders_ptr->end())
        {
            recv_ptr->emplace_back(message);
        }
    }

    return 1;
}


core::MessageEndpoint MessageBusCPUImpl::create_endpoint()
{
    const std::lock_guard lock(mutex_);

    using VT = std::vector<messaging::MessageVariant>;

    auto messages_to_send_v{std::make_shared<VT>()};
    auto recv_messages_v{std::make_shared<VT>()};

    auto endpoint = MessageEndpointCPU(std::make_shared<MessageEndpointCPUImpl>(messages_to_send_v, recv_messages_v));
    endpoint_data_.emplace_back(messages_to_send_v, recv_messages_v, endpoint.get_senders_ptr());
    return std::move(endpoint);
}

}  // namespace knp::core::messaging::impl
