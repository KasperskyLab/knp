/**
 * @file message_bus.cpp
 * @brief Message bus implementation.
 * @kaspersky_support Artiom N.
 * @date 21.02.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <knp/core/message_endpoint.h>

#include <message_endpoint_impl.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <memory>

// sleep_for.
#include <thread>

#include <boost/preprocessor.hpp>


namespace cm = knp::core::messaging;


namespace knp::core
{

UID MessageEndpoint::get_receiver_uid(const MessageEndpoint::SubscriptionVariant &subscription)
{
    return std::visit([](const auto &subscr) { return subscr.get_receiver_uid(); }, subscription);
}


messaging::MessageHeader get_header(const knp::core::messaging::MessageVariant &message)
{
    return std::visit([](const auto &msg) { return msg.header_; }, message);
}


bool operator<(const MessageEndpoint::SubscriptionVariant &sv1, const MessageEndpoint::SubscriptionVariant &sv2)
{
    return MessageEndpoint::get_receiver_uid(sv1) < MessageEndpoint::get_receiver_uid(sv2);
}


std::pair<size_t, UID> MessageEndpoint::get_subscription_key(const MessageEndpoint::SubscriptionVariant &subscription)
{
    return std::make_pair(subscription.index(), get_receiver_uid(subscription));
}


MessageEndpoint::MessageEndpoint(MessageEndpoint &&endpoint) noexcept
    : impl_(std::move(endpoint.impl_)),
      subscriptions_(std::move(endpoint.subscriptions_)),
      senders_(std::move(endpoint.senders_))
{
}


MessageEndpoint::~MessageEndpoint() = default;


template <typename MessageType>
Subscription<MessageType> &MessageEndpoint::subscribe(const UID &receiver, const std::vector<UID> &senders)
{
    constexpr size_t index = get_type_index<knp::core::messaging::MessageVariant, MessageType>;

    SPDLOG_DEBUG("Subscribing {} to the list of senders [{}], message type index = {}...",
        std::string(receiver), senders.size(),
        index);

    #if (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE)
    for (const auto &s_uid : senders) SPDLOG_TRACE("Sender UID = {}", std::string(s_uid));
    #endif

    auto iter = subscriptions_.find(std::make_pair(index, receiver));

    if (!senders_) senders_ = std::make_shared<std::unordered_set<knp::core::UID, knp::core::uid_hash>>();
    senders_->insert(senders.begin(), senders.end());

    if (iter != subscriptions_.end())
    {
        auto &sub = std::get<index>(iter->second);
        SPDLOG_TRACE("Existing subscription found, adding senders...");
        sub.add_senders(senders);
        return sub;
    }

    SPDLOG_TRACE("Existing subscription was not found, creating new subscription...");
    auto sub_variant = SubscriptionVariant{Subscription<MessageType>{receiver, senders}};
    assert(index == sub_variant.index());
    auto insert_res = subscriptions_.emplace(std::make_pair(index, receiver), sub_variant);
    auto &sub = std::get<index>(insert_res.first->second);
    return sub;
}


template <typename MessageType>
bool MessageEndpoint::unsubscribe(const UID &receiver)
{
    SPDLOG_DEBUG("Unsubscribing {}...", std::string(receiver));
    constexpr auto index = get_type_index<knp::core::messaging::MessageVariant, MessageType>;
    auto iter = subscriptions_.find(std::make_pair(index, receiver));
    if (iter != subscriptions_.end())
    {
        subscriptions_.erase(iter);
        return true;
    }
    update_senders();
    return false;
}


void MessageEndpoint::remove_receiver(const UID &receiver)
{
    SPDLOG_DEBUG("Removing receiver {}...", std::string(receiver));

    for (auto sub_iter = subscriptions_.begin(); sub_iter != subscriptions_.end(); ++sub_iter)
    {
        if (get_receiver_uid(sub_iter->second) == receiver)
        {
            subscriptions_.erase(sub_iter);
        }
    }
    update_senders();
}


void MessageEndpoint::send_message(const knp::core::messaging::MessageVariant &message)
{
    SPDLOG_TRACE(
        "Sending message from {}, index = {}...", std::string(get_header(message).sender_uid_), message.index());
    impl_->send_message(message);
}


bool MessageEndpoint::receive_message()
{
    SPDLOG_DEBUG("Receiving message...");

    auto message_opt = impl_->receive_message();
    if (!message_opt.has_value())
    {
        SPDLOG_TRACE("No message received.");
        return false;
    }
    auto &message = message_opt.value();
    const UID &sender_uid = get_header(message).sender_uid_;
    const size_t type_index = message.index();

    SPDLOG_TRACE("Subscription count = {}.", subscriptions_.size());

    // Find a subscription.
    for (auto &&[k, sub_variant] : subscriptions_)
    {
        if (sub_variant.index() != type_index)
        {
            SPDLOG_TRACE(
                "Subscription message type index does not match the message type index [{} != {}].",
                sub_variant.index(), type_index);
            continue;
        }

        std::visit(
            [&sender_uid, &message](auto &&subscription)
            {
                SPDLOG_TRACE("Adding message to subscription, checking sender UID: {}.", std::string(sender_uid));
                if (subscription.has_sender(sender_uid))
                {
                    SPDLOG_TRACE("Subscription has sender with UID {}.", std::string(sender_uid));
                    subscription.add_message(
                        std::get<typename std::decay_t<decltype(subscription)>::MessageType>(message));
                    SPDLOG_TRACE("Message with type index {} was added in the subscription to sender {}.",
                                 message.index(),
                                 std::string(sender_uid));
                }
                else
                {
                    SPDLOG_TRACE("Subscription has not sender UID: {}.", std::string(sender_uid));
                    SPDLOG_TRACE(
                        "Message was added to the subscription with receiver {}.",
                        std::string(get_receiver_uid(subscription)));
                }
            },
            sub_variant);

        // sub_variant.swap(v);
    }

    return true;
}


size_t MessageEndpoint::receive_all_messages(const std::chrono::milliseconds &sleep_duration)
{
    size_t messages_counter = 0;

    while (receive_message())
    {
        ++messages_counter;
        if (sleep_duration.count() != 0)
        {
            std::this_thread::sleep_for(sleep_duration);
        }
    }

    return messages_counter;
}


template <class MessageType>
std::vector<MessageType> MessageEndpoint::unload_messages(const knp::core::UID &receiver_uid)
{
    constexpr size_t index = get_type_index<knp::core::messaging::MessageVariant, MessageType>;
    auto iter = subscriptions_.find(std::make_pair(index, receiver_uid));

    if (iter == subscriptions_.end())
    {
        return {};
    }

    Subscription<MessageType> &subscription = std::get<index>(iter->second);
    auto result = std::move(subscription.get_messages());
    subscription.clear_messages();

    return result;
}


void MessageEndpoint::update_senders()
{
    if (!senders_) senders_ = std::make_shared<std::unordered_set<knp::core::UID, knp::core::uid_hash>>();

    std::unordered_set<knp::core::UID, knp::core::uid_hash> new_senders;
    new_senders.reserve(senders_->size());
    for (const auto &sub : subscriptions_)
    {
        auto sub_senders = std::visit([](auto &sub_var) { return sub_var.get_senders(); }, sub.second);
        new_senders.insert(sub_senders.begin(), sub_senders.end());
    }
}


#define INSTANCE_MESSAGES_FUNCTIONS(n, template_for_instance, message_type)                \
    template Subscription<cm::message_type> &MessageEndpoint::subscribe<cm::message_type>( \
        const UID &receiver, const std::vector<UID> &senders);                             \
    template bool MessageEndpoint::unsubscribe<cm::message_type>(const UID &receiver);     \
    template std::vector<cm::message_type> MessageEndpoint::unload_messages<cm::message_type>(const UID &receiver_uid);

BOOST_PP_SEQ_FOR_EACH(INSTANCE_MESSAGES_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_MESSAGES))

}  // namespace knp::core
