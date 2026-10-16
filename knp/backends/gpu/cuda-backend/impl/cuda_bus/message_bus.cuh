/**
 * @file message_bus.cuh
 * @brief CUDA message bus interface.
 * @kaspersky_support Vartenkov A.
 * @date 16.03.2025
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

#include <knp/core/uid.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/subscription.h>
#include <cub/config.cuh>

#include <cuda/std/variant>

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "subscription.cuh"
#include "../uid.cuh"
#include "messaging.cuh"
#include "../cuda_lib/vector.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief Structure for storing and finding a specific type of messages.
 * @tparam Message type of messages to store.
 */
template <class Message>
struct MessageBuffer
{
    std::vector<Message> messages_; // Owning CPU container, calls destructors
    std::unordered_map<knp::core::UID, std::vector<device_lib::LongIndex>, knp::core::uid_hash> message_ids_;

    void add_message(Message &&message_to_add)
    {
        // Critical section all.
        messages_.resize(messages_.size() + 1);
        messages_.back() = message_to_add;
        auto &msg = messages_.back();
        knp::core::UID uid = to_cpu_uid(msg.header_.sender_uid_);
        auto map_iter = message_ids_.find(uid);
        if (map_iter == message_ids_.end())
        {
            message_ids_.insert(std::make_pair(uid, std::vector<device_lib::LongIndex>{messages_.size() - 1}));
        }
        else
        {
            map_iter->second.push_back(messages_.size() - 1);
        }
    }

    [[nodiscard]] std::vector<device_lib::LongIndex> find_message_ids(const knp::core::UID &sender) const
    {
        auto iter = message_ids_.find(sender);
        if (iter == message_ids_.end())
            return {};
        return iter->second;
    }

    void clear()
    {
        messages_.clear();
        message_ids_.clear();
    }
};


/**
* @brief The MessageBus class is a definition of an interface to a message bus.
*/
class CUDAMessageBus
{
public:
    using CUDAMessageVariant = ::cuda::std::variant<SpikeMessage, SynapticImpactMessage>;

    /**
     * @brief Construct GPU message bus.
     * @param external_endpoint message endpoint used for message exchange with host.
     */
    explicit CUDAMessageBus(knp::core::MessageEndpoint &external_endpoint) : cpu_endpoint_{external_endpoint}
    {}

public:
    /**
     * @brief Add a subscription to messages of the specified type from senders with given UIDs.
     * @note If the subscription for the specified receiver and message type already exists, the method updates the list
     * of senders in the subscription.
     * @tparam MessageType type of messages to which the receiver subscribes via the subscription.
     * @param receiver receiver UID.
     * @param senders vector of sender UIDs.
     * @return true if a new subscription was created.
     */
    template <typename MessageType>
    __host__ bool subscribe_both(const cuda::UID &receiver, const std::vector<cuda::UID> &senders)
    {
        constexpr auto type_index = boost::mp11::mp_find<MessageVariant, MessageType>();
        return subscribe_both(receiver, senders, type_index);
    }

    template <typename MessageType>
    __host__ bool subscribe_gpu(const cuda::UID &receiver, const std::vector<cuda::UID> &senders)
    {
        constexpr auto type_index = boost::mp11::mp_find<MessageVariant, MessageType>();
        return subscribe_gpu(receiver, senders, type_index);
    }

    template <typename MessageType>
    [[nodiscard]] __host__ const std::vector<MessageType>& all_messages() const
    {
        return get_message_buffer<MessageType>().messages_;
    }

    /**
     * @brief Unsubscribe from messages of a specified type.
     * @tparam MessageType type of messages to which the receiver is subscribed.
     * @param receiver receiver UID.
     * @return true if a subscription was deleted, false otherwise.
     */
    template <typename MessageType>
    __host__ bool unsubscribe(const cuda::UID &receiver);

    /**
     * @brief Remove all subscriptions for a receiver with given UID.
     * @param receiver receiver UID.
     */
    __host__ void remove_receiver(const cuda::UID &receiver);

    template <class MessageType>
    __host__ void send_message(MessageType &&message)
    {
        get_message_buffer<MessageType>().add_message(std::move(message));
    }

    /**
     * @brief Delete all messages inside the bus.
     */
    __host__ void clear()
    {
        messages_spikes_.clear();
        messages_impacts_.clear();
    }

    template <class MessageType>
    __host__ void clear()
    {
        get_message_buffer<MessageType>().clear();
    }

    /**
     * @brief Copy host subscriptions here.
     */
    __host__ void sync_with_host();

    /**
     * @brief Receive messages from host.
     */
    __host__ void receive_messages_from_host();

    /**
     * @brief Send messages to host.
     */
    template <class MessageType>
    __host__ void send_messages_to_host(size_t step)
    {
        const MessageBuffer<MessageType> &msg_buffer = get_message_buffer<MessageType>();
        for (size_t i = 0; i < msg_buffer.messages_.size(); ++i)
        {
            const MessageType &msg = msg_buffer.messages_[i];
            const cuda::MessageHeader &header = msg.header_;
            if (header.send_time_ != step - 1 && header.send_time_ != step || header.is_external_) continue;
            auto host_message = make_host_message(msg);
            cpu_endpoint_.send_message(host_message);
        }
    }

    /**
     * @brief Send messages of the specified type to a bus.
     * @tparam MessageType type of messages to read.
     * @param receiver_uid receiver UID.
     * @return vector of messages.
     */
    template <class MessageType>
    __host__ void send_messages(const cuda::UID &receiver_uid, device_lib::CUDAVector<MessageType> &result_messages);

    template <class MessageType>
    __host__ std::vector<device_lib::LongIndex> unload_messages(const cuda::UID &receiver_uid) const
    {
        constexpr auto type_index = boost::mp11::mp_find<CUDAMessageVariant, MessageType>();
        size_t subscription_id = find_subscription(receiver_uid, type_index);
        Subscription sub = subscriptions_.copy_at(subscription_id);
        std::vector<device_lib::LongIndex> result;
        for (size_t i = 0; i < sub.get_senders().size(); ++i)
        {
            knp::core::UID uid = cuda::to_cpu_uid(sub.get_senders().copy_at(i));
            auto id_buf = get_message_buffer<MessageType>().find_message_ids(uid);
            auto res_size = result.size();
            result.resize(res_size + id_buf.size());
            // std::memcpy(result.data() + res_size, id_buf.data(), id_buf.size() * sizeof(device_lib::LongIndex));
            std::copy(id_buf.begin(), id_buf.end(), result.begin() + res_size);
        }
        return result;
    }

    template <class MessageType>
    __host__ size_t get_num_messages() const { return all_messages<MessageType>().size(); }

public:
    /**
     * @brief Type of subscription container.
     */
    using SubscriptionContainer = device_lib::CUDAVector<Subscription>;

    /**
     * @brief Get a reference of the subscription container of the endpoint.
     * @return reference to the subscription container.
     */
    SubscriptionContainer& get_subscriptions() { return subscriptions_; }

private:

    template<class MessageType>
    MessageBuffer<MessageType>& get_message_buffer();

    template<class MessageType>
    const MessageBuffer<MessageType>& get_message_buffer() const;

    /**
     * @brief Send messages to CPU endpoint.
     */
    __host__ int synchronize();

    __host__ void subscribe_host(const cuda::UID &receiver, const std::vector<cuda::UID> &senders, size_t type_id);

    __host__ bool subscribe_gpu(const cuda::UID &receiver, const std::vector<cuda::UID> &senders, size_t type_index)
    {
        SPDLOG_DEBUG("Looking for existing subscriptions");
        size_t sub_index = find_subscription(receiver, type_index);
        if (sub_index != subscriptions_.size())
        {
            Subscription sub_upd = subscriptions_.copy_at(sub_index);
            for (size_t i = 0; i < senders.size(); ++i)
            {
                sub_upd.add_sender(senders[i]);
            }
            subscriptions_.set(sub_index, sub_upd);
            return false;
        }
        SPDLOG_DEBUG("Adding new gpu subscription");
        subscriptions_.push_back(Subscription(receiver, senders, type_index));
        SPDLOG_DEBUG("Done adding new gpu subscription");
        return true;
    }

    __host__ bool subscribe_both(const cuda::UID &receiver, const std::vector<cuda::UID> &senders, size_t type_index)
    {
        subscribe_gpu(receiver, senders, type_index);
        subscribe_host(receiver, senders, type_index);
        SPDLOG_DEBUG("Done adding new host subscription");
        return true;
    }

    template <typename MessageType>
    [[nodiscard]] __host__ size_t find_subscription(const cuda::UID &receiver) const;

    [[nodiscard]] __host__ size_t find_subscription(const cuda::UID &receiver, size_t type_id) const;

    template <typename MessageType>
    __host__ __device__ ::cuda::std::vector<device_lib::LongIndex> find_messages(const Subscription &subscription);

    /**
     * @brief Container that stores all the subscriptions for the current endpoint.
     */
    SubscriptionContainer subscriptions_;
    device_lib::CUDAVector<cuda::UID> gpu_senders_;
    device_lib::CUDAVector<cuda::UID> host_senders_;

    MessageBuffer<SpikeMessage> messages_spikes_;
    MessageBuffer<SynapticImpactMessage> messages_impacts_;

    knp::core::MessageEndpoint &cpu_endpoint_;
};


template<>
inline MessageBuffer<SpikeMessage>& CUDAMessageBus::get_message_buffer<SpikeMessage>()
{
    return messages_spikes_;
}


template<>
inline MessageBuffer<SynapticImpactMessage>& CUDAMessageBus::get_message_buffer<SynapticImpactMessage>()
{
    return messages_impacts_;
}

template<>
inline const MessageBuffer<SpikeMessage>& CUDAMessageBus::get_message_buffer<SpikeMessage>() const
{
    return messages_spikes_;
}


template<>
inline const MessageBuffer<SynapticImpactMessage>& CUDAMessageBus::get_message_buffer<SynapticImpactMessage>() const
{
    return messages_impacts_;
}

} // namespace knp::backends::gpu::cuda
