/**
 * @file message_bus.h
 * @brief Message bus interface.
 * @kaspersky_support Artiom N.
 * @date 23.01.2023
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

#include <functional>
#include <memory>

/**
 * @brief Namespace for message bus implementations.
 */
namespace knp::core::messaging::impl
{
/**
 * @brief The MessageBusImpl class is an internal implementation class for message bus.
 */
class MessageBusImpl;
}  // namespace knp::core::messaging::impl


/**
 * @brief Core library namespace.
 */
namespace knp::core
{
/**
 * @brief The MessageBus class is a definition of an interface to a message bus.
 */
class MessageBus
{
public:
    /**
     * @brief Create a CPU-based message bus.
     * @return message bus.
     */
    static std::shared_ptr<MessageBus> construct_cpu_bus();

    /**
     * @brief Create a ZMQ-based message bus.
     * @return message bus.
     */
    static std::shared_ptr<MessageBus> construct_zmq_bus();

    /**
     * @brief Create a message bus with default implementation.
     * @return message bus.
     */
    static std::shared_ptr<MessageBus> construct_bus() { return construct_cpu_bus(); }

    /**
     * @brief Default message bus constructor is deleted.
     * @note Use one of the static functions above.
     */
    MessageBus() = delete;

    /**
     * @brief Move constructor.
     */
    MessageBus(MessageBus &&) noexcept;

    /**
     * @brief Message bus destructor.
     */
    ~MessageBus();

public:
    /**
     * @brief Create a new endpoint that sends and receives messages through the message bus.
     * @return new endpoint.
     * @see MessageEndpoint.
     */
    [[nodiscard]] MessageEndpoint create_endpoint();

    /**
     * @brief Route some messages.
     * @return number of messages routed during the step.
     */
    size_t step();

    /**
     * @brief Route messages.
     * @return number of messages routed.
     */
    size_t route_messages();

private:
    /**
     * @brief Message bus constructor with a specialized implementation.
     * @param impl message bus implementation.
     * @note Currently two implementations are available: ZMQ and CPU.
     */
    explicit MessageBus(std::unique_ptr<messaging::impl::MessageBusImpl> &&impl);


    /**
     * @brief Message bus implementation.
     */
    std::unique_ptr<messaging::impl::MessageBusImpl> impl_;
};

}  // namespace knp::core
