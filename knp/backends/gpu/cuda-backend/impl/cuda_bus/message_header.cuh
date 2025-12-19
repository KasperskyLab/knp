/**
 * @file message_header.cuh
 * @brief Message header class for CUDA.
 * @kaspersky_support Artiom N.
 * @date 02.04.2025
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

#include <knp/core/messaging/message_header.h>
#include <cstdint>
#include "../uid.cuh"


/**
 * @brief Messaging namespace.
 */
namespace  knp::backends::gpu::cuda
{
/**
 * @brief Common header for messages.
 */
class MessageHeader
{
public:
    /**
     * @brief UID of the object that sent the message.
     */
    cuda::UID sender_uid_;
    /**
     * @brief Index of the network execution step.
     */
    std::uint64_t send_time_;

    /**
     * @brief True if the message is converted from host message type.
     */
    bool is_external_ = false;

    __host__ __device__ bool operator==(const MessageHeader &other) const
    {
        return sender_uid_ == other.sender_uid_ && send_time_ == other.send_time_;
    }
};


namespace detail
{
inline cuda::MessageHeader make_gpu_message_header(const knp::core::messaging::MessageHeader &host_header)
{
    return {.sender_uid_ = to_gpu_uid(host_header.sender_uid_), .send_time_ = host_header.send_time_,
            .is_external_ = true};
}

inline knp::core::messaging::MessageHeader make_host_message_header(const cuda::MessageHeader &gpu_header)
{
    return {.sender_uid_ = to_cpu_uid(gpu_header.sender_uid_), .send_time_ = gpu_header.send_time_};
}
} // namespace detail
}  // namespace knp::backends::gpu::cuda
