/**
 * @file subscription.cu
 * @brief CUDA subscription implementation.
 * @kaspersky_support Artiom N.
 * @date 28.09.2025
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
#include <knp/core/messaging/message_envelope.h>

#include <unordered_set>
#include <variant>

#include "../cuda_lib/register_all.cuh"
#include "../cuda_lib/vector.cuh"
#include "subscription.cuh"

namespace knp::backends::gpu::cuda
{

__host__ Subscription::Subscription(const knp::core::MessageEndpoint::SubscriptionVariant &cpu_subscription)
{
    static_assert(
        ::cuda::std::variant_size<cuda::MessageVariant>() == std::variant_size<knp::core::messaging::MessageVariant>(),
        "This function requires 1-to-1 correspondence between host and GPU message types");

    const knp::core::MessageEndpoint::SubscriptionVariant &sub = cpu_subscription;
    size_t type_index = sub.index();
    core::UID receiver_uid =
        std::visit([](const auto &sub) { return sub.get_receiver_uid(); }, cpu_subscription);

    std::unordered_set<core::UID, core::uid_hash> senders = std::visit(
        [](const auto &sub)
        {
            UidSet result;
            for (const core::UID &sender : sub.get_senders()) result.insert(sender);
            return result;
        },
        cpu_subscription);
    receiver_ = receiver_uid;
    senders_ = senders;
    type_index_ = type_index;
}

}  // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
