/**
 * @file observer.h
 * @brief Observer class that unloads messages and stores them.
 * @kaspersky_support Vartenkov A.
 * @date 23.08.2023
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
#include <knp/core/impexp.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/mp11.hpp>


/**
 * @brief Monitoring namespace.
 */
namespace knp::framework::monitoring
{
/**
 * @brief Functor for message processing.
 * 
 * @tparam Message type of messages the functor processes.
 */
template <class Message>
using MessageProcessor = std::function<void(const std::vector<Message> &)>;


/**
 * @brief The MessageObserver class is a definition of an observer that receives messages and processes them.
 * 
 * @tparam Message message type that is processed by an observer.
 * 
 * @note Use this class for statistics calculation or for information output.
 */
template <class Message>
class MessageObserver
{
public:
    /**
     * @brief Constructor.
     * 
     * @param endpoint endpoint from which to get messages.
     * @param processor functor to process messages.
     * @param uid observer UID.
     */
    MessageObserver(
        core::MessageEndpoint &&endpoint, MessageProcessor<Message> &&processor, core::UID uid = core::UID{true})
        : endpoint_(std::move(endpoint)), process_messages_(processor), base_data_{uid}
    {
    }

    /**
     * @brief Avoid copy assignment of MessageObserver.
     */
    MessageObserver& operator=(const MessageObserver&) = delete;
    
    /**
     * @brief Move constructor for observer.
     * 
     * @param other other observer.
     */
    MessageObserver(MessageObserver<Message> &&other) noexcept = default;

    /**
     * @brief Subscribe to messages.
     * 
     * @param entities message senders.
     */
    void subscribe(const std::vector<core::UID> &entities) { endpoint_.subscribe<Message>(base_data_.uid_, entities); }

    /**
     * @brief Receive and process messages.
     */
    void update()
    {
        endpoint_.receive_all_messages();
        auto messages_raw = endpoint_.unload_messages<Message>(base_data_.uid_);
        process_messages_(messages_raw);
    }

    /**
     * @brief Get observer UID.
     * 
     * @return Observer UID.
     */
    [[nodiscard]] knp::core::UID get_uid() const { return base_data_.uid_; }

private:
    core::MessageEndpoint endpoint_;
    MessageProcessor<Message> process_messages_;
    core::BaseData base_data_;
};

/**
 * @brief List of all possible observers.
 */
using AllObservers = boost::mp11::mp_transform<MessageObserver, core::messaging::AllMessages>;

/**
 * @brief Observer variant that contains all possible observers.
 */
using AnyObserverVariant = boost::mp11::mp_rename<AllObservers, std::variant>;

}  // namespace knp::framework::monitoring
