/**
 * @file message_endpoint.h
 * @brief Message endpoint interface.
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

#include <knp/core/messaging/message_envelope.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/subscription.h>
#include <knp/core/uid.h>

#include <any>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/mp11.hpp>
#include <boost/noncopyable.hpp>


/**
 * @brief Namespace for message bus implementations.
 */
namespace knp::core::messaging::impl
{
/**
 * @brief The MessageEndpointImpl class is an internal implementation class for message endpoint.
 */
class MessageEndpointImpl;
}  // namespace knp::core::messaging::impl


namespace knp::core
{
/**
 * @class boost::noncopyable
 * @brief Ensure that objects cannot be copied.
 * @see <a href="https://www.boost.org/doc/libs/1_80_0/boost/core/noncopyable.hpp">boost::noncopyable description</a>
 */

/**
 * @brief The MessageEndpoint class is a definition of message endpoints.
 * @details You can use message endpoints to receive or send messages.
 */
class MessageEndpoint : private boost::noncopyable
{
public:
    /**
     * @brief List of subscription types based on message types specified in `messaging::AllMessages`.
     */
    using AllSubscriptions = boost::mp11::mp_transform<Subscription, messaging::AllMessages>;

    /**
     * @brief Subscription variant that contains any subscription type specified in `AllSubscriptions`.
     * @details `SubscriptionVariant` takes the value of `std::variant<SubscriptionType_1,..., SubscriptionType_n>`,
     * where `SubscriptionType_[1..n]` is the subscription type specified in `AllSubscriptions`. \n For example, if
     * `AllSubscriptions` contains SpikeMessage and SynapticImpactMessage types, then `SubscriptionVariant =
     * std::variant<SpikeMessage, SynapticImpactMessage>`. \n `SubscriptionVariant` retains the same order of message
     * types as defined in `AllSubscriptions`.
     * @see ALL_MESSAGES.
     */
    using SubscriptionVariant = boost::mp11::mp_rename<AllSubscriptions, std::variant>;

public:
    /**
     * @brief Get receiver UID from a subscription variant.
     * @param subscription subscription variant.
     * @return receiver UID.
     */
    static UID get_receiver_uid(const SubscriptionVariant &subscription);
    /**
     * @brief Get subscription key from a subscription variant.
     * @param subscription subscription variant.
     * @return pair of subscription index and subscription key.
     */
    static std::pair<size_t, UID> get_subscription_key(const SubscriptionVariant &subscription);

    /**
     * @brief Find index of an entity type in its variant.
     * @details For example, you can use the method to find an index of a message type in a message variant or an index
     * of a subscription type in a subscription variant.
     * @tparam Variant variant of one or more entity types.
     * @tparam Type entity type to search.
     */
    template <typename Variant, typename Type>
    static constexpr size_t get_type_index = boost::mp11::mp_find<Variant, Type>::value;

public:
    /**
     * @brief Move constructor for message endpoints.
     * @param endpoint endpoint to move.
     */
    MessageEndpoint(MessageEndpoint &&endpoint) noexcept;

    /**
     * @brief Avoid copy assignment of MessageEndpoint.
     */
    MessageEndpoint &operator=(const MessageEndpoint &) = delete;

    /**
     * @brief Message endpoint destructor.
     */
    virtual ~MessageEndpoint();

public:
    /**
     * @brief Add a subscription to messages of the specified type from senders with given UIDs.
     * @note If the subscription for the specified receiver and message type already exists, the method updates the list
     * of senders in the subscription.
     * @tparam MessageType type of messages to which the receiver subscribes via the subscription.
     * @param receiver receiver UID.
     * @param senders vector of sender UIDs.
     * @return number of senders added to the subscription.
     */

    template <typename MessageType>
    Subscription<MessageType> &subscribe(const UID &receiver, const std::vector<UID> &senders);

    /**
     * @brief Unsubscribe from messages of a specified type.
     * @tparam MessageType type of messages to which the receiver is subscribed.
     * @param receiver receiver UID.
     * @return true if a subscription was deleted, false otherwise.
     */
    template <typename MessageType>
    bool unsubscribe(const UID &receiver);

    /**
     * @brief Remove all subscriptions for a receiver with given UID.
     * @param receiver receiver UID.
     */
    void remove_receiver(const UID &receiver);

    /**
     * @brief Send a message to the message bus.
     * @param message message to send.
     */
    void send_message(const knp::core::messaging::MessageVariant &message);

    /**
     * @brief Receive a message from the message bus.
     * @return `true` if a message was received, `false` if no message was received.
     */
    bool receive_message();

    /**
     * @brief Receive all messages that were sent to the endpoint.
     * @param sleep_duration time interval in milliseconds between the moments of receiving messages.
     * @return number of received messages.
     */
    size_t receive_all_messages(const std::chrono::milliseconds &sleep_duration = std::chrono::milliseconds(0));

    /**
     * @brief Read messages of the specified type received via subscription.
     * @note After reading the messages, the method clears them from the subscription.
     * @tparam MessageType type of messages to read.
     * @param receiver_uid receiver UID.
     * @return vector of messages.
     */
    template <class MessageType>
    std::vector<MessageType> unload_messages(const knp::core::UID &receiver_uid);

public:
    /**
     * @brief Type of subscription container.
     */
    using SubscriptionContainer = std::map<std::pair<size_t, UID>, SubscriptionVariant>;

    /**
     * @brief Get access to subscription container of the endpoint.
     * @return Reference to subscription container.
     */
    const SubscriptionContainer &get_endpoint_subscriptions() const { return subscriptions_; }

    /**
     * @brief Get list of senders.
     * @return weak pointer to unordered set of sender UIDs.
     */
    auto get_senders_ptr()
    {
        std::weak_ptr<std::unordered_set<knp::core::UID, knp::core::uid_hash>> result{senders_};
        return result;
    }

protected:
    /**
     * @brief Message endpoint implementation.
     */
    std::shared_ptr<messaging::impl::MessageEndpointImpl> impl_;

protected:
    /**
     * @brief Message endpoint default constructor.
     */
    MessageEndpoint() = default;

private:
    /**
     * @brief Container that stores all the subscriptions for the current endpoint.
     */
    SubscriptionContainer subscriptions_;

    /**
     * @brief A set that contains all senders that this endpoint receives messages from.
     */
    std::shared_ptr<std::unordered_set<knp::core::UID, knp::core::uid_hash>> senders_ =
        std::make_shared<std::unordered_set<knp::core::UID, knp::core::uid_hash>>();

    /**
     * @brief Update list of senders.
     */
    void update_senders();
};

}  // namespace knp::core
