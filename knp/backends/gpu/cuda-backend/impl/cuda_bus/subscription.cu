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
#include "subscription.cuh"
#include "../cuda_lib/vector.cuh"

#include <variant>
#include <knp/core/messaging/message_envelope.h>

namespace knp::backends::gpu::cuda
{

__host__ Subscription::Subscription(const knp::core::MessageEndpoint::SubscriptionVariant &cpu_subscription)
{
    static_assert(::cuda::std::variant_size<cuda::MessageVariant>()
                  == std::variant_size<knp::core::messaging::MessageVariant>(),
                  "This function requires 1-to-1 correspondence between host and GPU message types");

    const knp::core::MessageEndpoint::SubscriptionVariant &sub = cpu_subscription;
    size_t type_index = sub.index();
    cuda::UID receiver_uid = std::visit([](const auto &sub)
        {
            return cuda::to_gpu_uid(sub.get_receiver_uid());
        }, cpu_subscription);

    std::vector<cuda::UID> senders = std::visit([](const auto &sub)
        {
            std::vector<cuda::UID> result;
            result.reserve(sub.get_senders().size());
            for (const knp::core::UID &sender : sub.get_senders())
                result.push_back(cuda::to_gpu_uid(sender));
            return result;
        }, cpu_subscription);
    receiver_ = receiver_uid;
    senders_ = senders;
    type_index_ = type_index;
}

} // namespace knp::backends::gpu::cuda
