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

#include <boost/mp11.hpp>
#include <cuda/std/iterator>
#include <spdlog/spdlog.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <vector>
#include <knp/core/message_endpoint.h>

#include "../cuda_lib/printf.cuh"
#include "../cuda_lib/vector.cuh"
#include "../cuda_lib/safe_call.cuh"
#include "../cuda_lib/kernels.cuh"
#include "../cuda_lib/extraction.cuh"
#include "../cuda_lib/get_blocks_config.cuh"
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
    using UidSet = device_lib::CUDAVector<UID>;

public:
    Subscription() = default;

    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    __host__ Subscription(const cuda::UID &receiver, const std::vector<cuda::UID> &senders, int type_index) :
            receiver_(receiver), type_index_(type_index)
    {
        SPDLOG_TRACE("Initializing with {} senders.", senders.size());
        for (size_t i = 0; i < senders.size(); ++i) add_sender(senders[i]);
        SPDLOG_TRACE("Created a subscription with {} senders.", senders_.size());
    }

    __host__ Subscription(const knp::core::MessageEndpoint::SubscriptionVariant &cpu_subscription);

    __device__ Subscription(const cuda::UID &receiver,
                            const device_lib::CUDAVector<cuda::UID> &senders,
                            int type_index) :
    receiver_(receiver), type_index_(type_index)
    {
        for (size_t i = 0; i < senders.size(); ++i)
        {
            add_sender(*(senders.data() + i));
        }
    }

    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    __host__ Subscription(const gpu::cuda::UID &receiver, const std::vector<gpu::cuda::UID> &senders) :
        receiver_(receiver)
    {
        SPDLOG_TRACE("Initializing with {} senders.", senders.size());
        for (const auto &sender : senders) add_sender(sender);
        SPDLOG_TRACE("Created a subscription with {} senders.", senders_.size());
    }

    /**
     * @brief Get list of sender UIDs.
     * @return senders UIDs.
     */
    [[nodiscard]] __device__ __host__ const UidSet &get_senders() const { return senders_; }

    /**
     * @brief get message type for subscription.
     */
     [[nodiscard]] __device__ __host__ int type() const { return type_index_; }

    /**
      * @brief Get UID of the entity that receives messages via the subscription.
      * @return UID.
      */
    [[nodiscard]] __device__ __host__ cuda::UID get_receiver_uid() const { return receiver_; }

    /**
     * @brief Unsubscribe from a sender.
     * @details If a sender is not associated with the subscription, the method doesn't do anything.
     * @param uid sender UID.
     * @return true if sender was deleted from subscription.
     */
    __device__ __host__ bool remove_sender(const cuda::UID &uid)
    {
        for (uint64_t index = 0; index < senders_.size(); ++index)
        {
            if (senders_.copy_at(index) == uid)
            {
                auto iter = senders_.begin() + index;
                senders_.erase(iter, iter + 1);
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Add a sender with the given UID to the subscription.
     * @details If a sender is already associated with the subscription, the method doesn't do anything.
     * @param uid UID of the new sender.
     * @return true if sender added.
     */
    __host__ __device__ bool add_sender(const cuda::UID &uid)
    {
        if (has_sender(uid)) return false;
        senders_.push_back(uid);
        return true;
    }

    /**
     * @brief Add several senders to the subscription.
     * @param senders vector of sender UIDs.
     * @return number of senders added.
     */
    __host__ __device__ size_t add_senders(const device_lib::CUDAVector<cuda::UID> &senders)
    {
        size_t result = 0;
        for (size_t i = 0; i < senders.size(); ++i)
        {
            result += add_sender(senders.copy_at(i));
        }
        return result;
    }

    /**
     * @brief Check if a sender with the given UID exists.
     * @param uid sender UID.
     * @return `true` if the sender with the given UID exists, `false` if the sender with the given UID doesn't exist.
     */
    [[nodiscard]] __host__ __device__ bool has_sender(const cuda::UID &uid) const
    {
#if defined(__CUDA_ARCH__)
        PRINTF_TRACE("Using has_sender on device\n");
        for (size_t i = 0; i < senders_.size(); ++i)
        {
            if (senders_[i] == uid)
            {
                PRINTF_TRACE("Found sender\n");
                return true;
            }
        }
        PRINTF_TRACE("No sender found\n");
        return false;
#else
        if (senders_.size() == 0) return false;
        int *d_result = nullptr;
        cudaMalloc(&d_result, sizeof(int));
        cudaMemset(d_result, 0, sizeof(int));
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(senders_.size());
        const UID *senders_data = senders_.data();
        device_lib::has_sender_kernel<<<num_blocks, num_threads>>>(uid, senders_data, senders_.size(), d_result);
        cudaDeviceSynchronize();
        int result;
        cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return result;
#endif
    }

    __host__ __device__ bool is_my_message(const MessageVariant &message)
    {
        int message_type = message.index();
        if (message_type != type_index_) return false;

        UID message_sender = ::cuda::std::visit([](const auto &msg) { return msg.header_.sender_uid_; },
                                                message);
        return has_sender(message_sender);
    }

    __host__ __device__ bool operator==(const Subscription &other) const
    {
        return receiver_ == other.receiver_ && senders_ == other.senders_;
    }

    __host__ __device__ bool operator!=(const Subscription &other) const
    {
        return !(*this == other);
    }

public:
    /**
     * @brief Restore after shallow copying from device.
     */
    __host__ __device__ void actualize()
    {
    #ifndef __CUDA_ARCH__
        SPDLOG_TRACE("Actualizing subscription senders");
    #endif
        senders_.actualize();
    }

private:
    /**
     * @brief Receiver UID.
     */
    cuda::UID receiver_;

    /**
     * @brief Set of sender UIDs.
     */
    UidSet senders_;

    /**
     * @brief message type index.
     */
    int type_index_;
};

} // namespace knp::backends::gpu::cuda
