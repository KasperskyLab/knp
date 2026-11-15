/**
 * @file message_endpoint_zmq_impl.h
 * @brief Message endpoint ZeroMQ implementation header.
 * @kaspersky_support Artiom N.
 * @date 31.03.2023
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
#include <message_endpoint_impl.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <vector>

#include <zmq.hpp>


/**
 * @brief Namespace for implementations of message bus.
 */
namespace knp::core::messaging::impl
{

class MessageEndpointZMQImpl : public MessageEndpointImpl
{
public:
    explicit MessageEndpointZMQImpl(zmq::socket_t &&sub_socket, zmq::socket_t &&pub_socket);

public:
    std::optional<messaging::MessageVariant> receive_message() override
    {
        auto message_var = receive_zmq_message();
        if (!message_var.has_value())
        {
            return {};
        }

        auto message = knp::core::messaging::extract_from_envelope(message_var->data(), message_var->size());
        return message;
    }

    void send_message(const knp::core::messaging::MessageVariant &message) override
    {
        auto packed_msg = knp::core::messaging::pack_to_envelope(message);
        SPDLOG_TRACE("Packed message size: {}.", packed_msg.size());
        send_zmq_message(packed_msg.data(), packed_msg.size());
    }

public:
    void send_zmq_message(const std::vector<uint8_t> &data);
    void send_zmq_message(const void *data, size_t size);
    std::optional<zmq::message_t> receive_zmq_message();

private:
    // zmq::context_t &context_;
    zmq::socket_t sub_socket_;
    zmq::socket_t pub_socket_;
};

}  // namespace knp::core::messaging::impl
