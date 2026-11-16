/**
 * @file subscription.cuh
 * @brief Subscription class that determines message exchange between entities in the network.
 * @kaspersky_support
 * @date 15.03.2025
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

#include <knp/core/message_endpoint.h>

#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

#include <algorithm>
#include <cuda/std/iterator>
#include <vector>

#include <boost/mp11.hpp>

#include "../cuda_lib/extraction.cuh"
#include "../cuda_lib/get_blocks_config.cuh"
#include "../cuda_lib/kernels.cuh"
#include "../cuda_lib/printf.cuh"
#include "../cuda_lib/safe_call.cuh"
#include "../cuda_lib/vector.cuh"
#include "../uid.cuh"
#include "messaging.cuh"


namespace knp::backends::gpu::cuda
{
/**
 * @brief The Subscription class is used for message exchange between the network entities.
 * @tparam MessageT type of messages that are exchanged via the subscription.
 */
class Subscription final
{
public:
    /**
     * @brief Internal container for UIDs.
     */
    using UidSet = std::unordered_set<core::UID, core::uid_hash>;

public:
    Subscription() = default;

    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    __host__ Subscription(const core::UID &receiver, const std::vector<core::UID> &senders, int type_index)
        : receiver_(receiver), type_index_(type_index)
    {
        SPDLOG_TRACE("Initializing with {} senders.", senders.size());
        for (size_t i = 0; i < senders.size(); ++i) add_sender(senders[i]);
        SPDLOG_TRACE("Created a subscription with {} senders.", senders_.size());
    }

    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    __host__ Subscription(const core::UID &receiver, const std::unordered_set<core::UID, core::uid_hash> &senders,
                          int type_index)
                          : receiver_(receiver), type_index_(type_index)
    {
        SPDLOG_TRACE("Initializing with {} senders.", senders.size());
        senders_ = senders;
        SPDLOG_TRACE("Created a subscription with {} senders.", senders_.size());
    }

    /**
     * @brief Convert a CPU subscription to a GPU one.
     * @param cpu_subscription CPU-based subscription.
     */
    __host__ Subscription(const knp::core::MessageEndpoint::SubscriptionVariant &cpu_subscription);

    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    __host__ Subscription(const core::UID &receiver, const std::vector<core::UID> &senders)
        : receiver_(receiver)
    {
        SPDLOG_TRACE("Initializing with {} senders.", senders.size());
        for (const auto &sender : senders) add_sender(sender);
        SPDLOG_TRACE("Created a subscription with {} senders.", senders_.size());
    }

    /**
     * @brief Get list of sender UIDs.
     * @return senders UIDs.
     */
    [[nodiscard]] __host__ const UidSet &get_senders() const { return senders_; }

    /**
     * @brief get message type for subscription.
     */
    [[nodiscard]] __host__ int type() const { return type_index_; }

    /**
     * @brief Get UID of the entity that receives messages via the subscription.
     * @return UID.
     */
    [[nodiscard]] __host__ core::UID get_receiver_uid() const { return receiver_; }

    /**
     * @brief Unsubscribe from a sender.
     * @details If a sender is not associated with the subscription, the method doesn't do anything.
     * @param uid sender UID.
     * @return true if sender was deleted from subscription.
     */
    __host__ bool remove_sender(const core::UID &uid)
    {
        return senders_.erase(uid);
    }

    /**
     * @brief Add a sender with the given UID to the subscription.
     * @details If a sender is already associated with the subscription, the method doesn't do anything.
     * @param uid UID of the new sender.
     * @return true if sender added.
     */
    __host__ bool add_sender(const core::UID &uid)
    {
        if (has_sender(uid)) return false;
        senders_.insert(uid);
        return true;
    }

    /**
     * @brief Add several senders to the subscription.
     * @param senders vector of sender UIDs.
     * @return number of senders added.
     */
    __host__ size_t add_senders(const std::vector<core::UID> &senders)
    {
        size_t result = 0;
        for (const auto &uid : senders)
        {
            bool successfully_inserted = senders_.insert(uid).second;
            result += static_cast<size_t>(successfully_inserted);
        }
        return result;
    }

    /**
     * @brief Check if a sender with the given UID exists.
     * @param uid sender UID.
     * @return `true` if the sender with the given UID exists, `false` if the sender with the given UID doesn't exist.
     */
    [[nodiscard]] __host__ bool has_sender(const core::UID &uid) const
    {
        if (senders_.size() == 0) return false;
        return senders_.find(uid) != senders_.end();
    }

    /**
     * @brief Check a message if it corresponds to the current subscription.
     * @param message the message in question.
     * @return true if the message has the correct type and its sender is in this subscription sender UID list.
     */
    __host__ bool is_my_message(const MessageVariant &message)
    {
        int message_type = message.index();
        if (message_type != type_index_) return false;

        core::UID message_sender = ::cuda::std::visit([](const auto &msg) { return msg.header_.sender_uid_; }, message);
        return has_sender(message_sender);
    }

    /**
     * @brief Check if two subscriptions are equal.
     * @param other the other subscription.
     * @return true if the subscriptions are equal.
     */
    __host__ bool operator==(const Subscription &other) const
    {
        return receiver_ == other.receiver_ && senders_ == other.senders_ && type_index_ == other.type_index_;
    }

    __host__ bool operator!=(const Subscription &other) const
    {
        return !(*this == other);
    }


private:
    /**
     * @brief Receiver UID.
     */
    core::UID receiver_;

    /**
     * @brief Set of sender UIDs.
     */
    UidSet senders_;

    /**
     * @brief message type index.
     */
    int type_index_;
};

}  // namespace knp::backends::gpu::cuda
