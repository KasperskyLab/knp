/**
 * @file message_bus.cpp
 * @brief Message bus implementation.
 * @kaspersky_support Artiom N.
 * @date 21.02.2023
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

#include <knp/core/message_bus.h>
#include <knp/meta/macro.h>

#include <spdlog/spdlog.h>

#include <zmq.hpp>

#include "message_bus_cpu_impl/message_bus_cpu_impl.h"
#include "message_bus_zmq_impl/message_bus_zmq_impl.h"


namespace knp::core
{
MessageBus::~MessageBus() = default;

MessageBus::MessageBus(MessageBus &&) noexcept = default;

std::shared_ptr<MessageBus> MessageBus::construct_cpu_bus()
{
    return std::shared_ptr<MessageBus>(new MessageBus(std::make_unique<messaging::impl::MessageBusCPUImpl>()));
}


std::shared_ptr<MessageBus> MessageBus::construct_zmq_bus()
{
    return std::shared_ptr<MessageBus>(new MessageBus(std::make_unique<messaging::impl::MessageBusZMQImpl>()));
}


MessageBus::MessageBus(std::unique_ptr<messaging::impl::MessageBusImpl> &&impl) : impl_(std::move(impl))
{
    if (!impl_)
    {
        throw std::runtime_error("Unavailable message bus implementation.");
    }
}


MessageEndpoint MessageBus::create_endpoint()
{
    return impl_->create_endpoint();
}


size_t MessageBus::step()
{
    return impl_->step();
}


size_t MessageBus::route_messages()
{
    SPDLOG_DEBUG("Message routing cycle started.");
    size_t count = 0;
    impl_->update();
    size_t num_messages = step();

    KNP_UNROLL_LOOP()
    while (num_messages != 0)
    {
        count += num_messages;
        num_messages = step();
    }

    return count;
}

}  // namespace knp::core
