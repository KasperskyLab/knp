/**
 * @file message_bus.cuh
 * @brief CUDA message bus interface.
 * @kaspersky_support Artiom N.
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
 * @brief The MessageBus class is a definition of an interface to a message bus.
 */
class CUDAMessageBus
{
public:
    /**
     * @brief Construct GPU message bus.
     * @param external_endpoint message endpoint used for message exchange with host.
     */
    explicit CUDAMessageBus(knp::core::MessageEndpoint &external_endpoint) :
        cpu_endpoint_{external_endpoint}
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

    [[nodiscard]] __host__ const device_lib::CUDAVector<MessageVariant> & all_messages() const
    {
        return messages_to_route_;
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

    /**
     * @brief Send a message to the message bus.
     * @param message message to send.
     */
    __host__ __device__ void send_message(const cuda::MessageVariant &message);

    /**
     * @brief Send a batch of messages from a gpu pointer to message vector.
     * @param vec gpu pointer to message vector.
     */
    __host__ void send_message_gpu_batch(const device_lib::CUDAVector<cuda::MessageVariant> &vec);

    /**
     * @brief Delete all messages inside the bus.
     */
    __host__ void clear() { messages_to_route_.clear(); }

    /**
     * @brief Reserve bus buffer for messages.
     * @param num_messages number of messages.
     */
    __host__ void reserve_message_buffer(uint64_t num_messages) { messages_to_route_.reserve(num_messages); }

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
    __host__ void send_messages_to_host(size_t step);

    /**
     * @brief Send messages of the specified type to a bus.
     * @tparam MessageType type of messages to read.
     * @param receiver_uid receiver UID.
     * @return vector of messages.
     */
    template <class MessageType>
    __host__ void send_messages(const cuda::UID &receiver_uid, device_lib::CUDAVector<MessageType> &result_messages);

    template <class MessageType>
    __device__ const MessageType& get_message_gpu(size_t message_index) const
    {
        return ::cuda::std::get<MessageType>(messages_to_route_[message_index]);
    }

    template<class MessageType>
    __host__ MessageType get_message_cpu(size_t message_index) const
    {
        return ::cuda::std::get<MessageType>(messages_to_route_.copy_at(message_index));
    }

    template <class MessageType>
    __host__ device_lib::CUDAVector<uint64_t> unload_messages(const cuda::UID &receiver_uid);

    __host__ size_t get_num_messages() { return messages_to_route_.size(); }


public:
    /**
     * @brief Type of subscription container.
     */
    using SubscriptionContainer = device_lib::CUDAVector<Subscription>;

    using MessageBuffer = device_lib::CUDAVector<cuda::MessageVariant>;

    /**
     * @brief Get a reference of the subscription container of the endpoint.
     * @return reference to the subscription container.
     */
    SubscriptionContainer& get_subscriptions() { return subscriptions_; }

private:
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
    __host__ size_t find_subscription(const cuda::UID &receiver);

    __host__ size_t find_subscription(const cuda::UID &receiver, size_t type_id);

    template <typename MessageType>
    __host__ __device__ ::cuda::std::vector<uint64_t> find_messages(const Subscription &subscription);

    /**
     * @brief Container that stores all the subscriptions for the current endpoint.
     */
    SubscriptionContainer subscriptions_;
    device_lib::CUDAVector<cuda::UID> gpu_senders_;
    device_lib::CUDAVector<cuda::UID> host_senders_;
    MessageBuffer messages_to_route_;
    knp::core::MessageEndpoint &cpu_endpoint_;
};

}  // namespace knp::backends::gpu::cuda
