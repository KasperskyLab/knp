/**
 * @file message_bus.cu
 * @brief Message bus implementation.
 * @kaspersky_support Vartenkov A.
 * @date 25.06.2026
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

#include <algorithm>
#include <vector>
#include <boost/mp11/algorithm.hpp>
#include <boost/preprocessor.hpp>
#include <cuda/std/detail/libcxx/include/algorithm>
#include <thrust/device_ptr.h>
#include <knp/meta/macro.h>
#include "message_bus.cuh"

#include "../cuda_lib/vector.cuh"
#include "../cuda_lib/register_all.cuh"
#include "../cuda_lib/get_blocks_config.cuh"


namespace knp::backends::gpu::cuda
{
template <class T>
using DevVec = device_lib::CUDAVector<T>;


__global__ void find_subscription_by_receiver(const Subscription *subscriptions, size_t size, const UID receiver,
                                              size_t type, size_t *index_out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const Subscription &sub = subscriptions[i];
    if (sub.type() != type) return;
    if (sub.get_receiver_uid() == receiver) *index_out = i; // Should only work once, so no race condition problems.
}


template <typename MessageType>
__host__ size_t CUDAMessageBus::find_subscription(const cuda::UID &receiver) const
{
    constexpr size_t type_index = boost::mp11::mp_find<MessageVariant, MessageType>::value;
    return find_subscription(receiver, type_index);
}


__host__ size_t CUDAMessageBus::find_subscription(const cuda::UID &receiver, size_t type_index) const
{
    if (!subscriptions_.size()) return 0;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(subscriptions_.size());

    size_t *index_out;
    cudaMalloc(&index_out, sizeof(size_t));
    size_t subscriptions_size = subscriptions_.size();
    cudaMemcpy(index_out, &subscriptions_size, sizeof(size_t), cudaMemcpyHostToDevice);

    find_subscription_by_receiver<<<num_blocks, num_threads>>>(subscriptions_.data(), subscriptions_.size(), receiver,
                                                               type_index, index_out);

    size_t result;
    cudaMemcpy(&result, index_out, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(index_out);
    return result;
}


template <typename MessageType>
__host__ bool CUDAMessageBus::unsubscribe(const cuda::UID &receiver)
{
    SPDLOG_DEBUG("Unsubscribing");
    size_t sub_index = find_subscription<MessageType>(receiver);
    if (sub_index >= subscriptions_.size())
    {
        SPDLOG_TRACE("No subscriptions found to unsubscribe from: returned {}", sub_index);
        return false;
    }
    SPDLOG_TRACE("Removing subscription #{}", sub_index);
    subscriptions_.erase(subscriptions_.begin() + sub_index, subscriptions_.begin() + sub_index + 1);
    SPDLOG_TRACE("Done unsubscribing");
    return true;
}


__host__ void CUDAMessageBus::remove_receiver(const cuda::UID &receiver)
{
    for (auto sub_iter = subscriptions_.begin(); sub_iter != subscriptions_.end(); ++sub_iter)
    {
        // TODO: Finish
    }
}


/**
* @brief Find all message indices that correspond to the current subscription.
* @param messages pointer to messages
* @param messages_size number of messages
* @param subscription the subscription used for searching
* @param indexes
* @param counter a zero-initialized counter, it would be equal to number of found messages after the function finishes
* @return
*/
__global__ void find_messages_kernel(const MessageVariant *messages, size_t messages_size, Subscription *subscription,
                                     device_lib::LongIndex *indices, device_lib::LongIndex *counter)
{
    const device_lib::LongIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    PRINTF_DEBUG("Find message kernel, i %lu\n", i);
    if (i >= messages_size) return;
    if (subscription->is_my_message(messages[i]))
    {
        device_lib::LongIndex index = atomicAdd(counter, 1ull);
        PRINTF_DEBUG("Found message: index %lu, message_index %lu\n", i, index);
        indices[index] = i;
    }
    else
        PRINTF_DEBUG("No message found!\n");
}


__global__ void get_message_kernel(const MessageVariant *var, int *type, const void **msg) {
    int type_val = var->index();
    switch (type_val)
    {
        // TODO : Add more after adding new messages
        static_assert(::cuda::std::variant_size<cuda::MessageVariant>() == 2, "Add a case statement here!");
        case 0:
            *msg = ::cuda::std::get_if<0>(var);
            break;
        case 1:
            *msg = ::cuda::std::get_if<1>(var);
            break;
        default:
            *msg = nullptr;
    }
    *type = type_val;
}


__host__ void CUDAMessageBus::subscribe_host(const cuda::UID &receiver, const std::vector<cuda::UID> &senders,
                                             size_t type_id)
{
    const knp::core::UID host_receiver = to_cpu_uid(receiver);
    std::vector <knp::core::UID> host_senders;
    host_senders.reserve(senders.size());
    for (const cuda::UID &cuda_uid : senders)
    {
        host_senders.push_back(to_cpu_uid(cuda_uid));
    }
    // TODO: Do it in a normal way maybe.
    static_assert(::cuda::std::variant_size<cuda::MessageVariant>() == 2, "Add a case statement here!");
    switch (type_id)
    {
        case 0:
            cpu_endpoint_.subscribe < boost::mp11::mp_at_c < knp::core::messaging::MessageVariant, 0 >> (
                    host_receiver, host_senders);
            break;
        case 1:
            cpu_endpoint_.subscribe < boost::mp11::mp_at_c < knp::core::messaging::MessageVariant, 1 >> (
                    host_receiver, host_senders);
            break;
    }
}


// We need to check that the messages we get from host were not previously sent there by GPU.
__global__ void same_sender_kernel(cuda::UID uid, cuda::Subscription *subscriptions, size_t sub_size,
                                   bool *result)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sub_size) return;
    if (subscriptions[i].get_receiver_uid() == uid) *result = true;
}


__host__ bool same_sender(const knp::core::messaging::MessageVariant &message,
                          CUDAMessageBus::SubscriptionContainer &subs)
{
    if (subs.size() == 0) return false;

    const knp::core::UID sender_uid = std::visit([](const auto &msg) { return msg.header_.sender_uid_; }, message);

    const cuda::UID gpu_uid = to_gpu_uid(sender_uid);
    bool result = false;
    bool *result_ptr;

    call_and_check(cudaMalloc(&result_ptr, sizeof(bool)));
    call_and_check(cudaMemcpy(result_ptr, &result, sizeof(bool), cudaMemcpyHostToDevice));
    const auto [num_blocks, num_threads] = device_lib::get_blocks_config(subs.size());
    same_sender_kernel<<<num_blocks, num_threads>>>(gpu_uid, subs.data(), subs.size(), result_ptr);
    call_and_check(cudaMemcpy(&result, result_ptr, sizeof(bool), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(result_ptr));
    return result;
}


/**
* @brief Copy host subscriptions here.
*/
__host__ void CUDAMessageBus::sync_with_host()
{
    int device;
    cudaGetDevice(&device);
    SPDLOG_DEBUG("Current CUDA device: {}", device);
    for (const auto &cpu_subscription : cpu_endpoint_.get_endpoint_subscriptions())
    {
        cuda::Subscription gpu_sub{cpu_subscription.second};
        std::vector<cuda::UID> senders;
        const auto &gpu_senders = gpu_sub.get_senders();
        senders.resize(gpu_senders.size());
        cudaMemcpy(senders.data(), gpu_senders.data(), sizeof(cuda::UID) * gpu_senders.size(), cudaMemcpyDeviceToHost);
        subscribe_both(gpu_sub.get_receiver_uid(), senders, gpu_sub.type());
    }
}


// TODO: Maybe template this or something.
__host__ void CUDAMessageBus::receive_messages_from_host()
{
    cpu_endpoint_.receive_all_messages();
    SPDLOG_DEBUG("CPU subscriptions: {}", cpu_endpoint_.get_endpoint_subscriptions().size());
    for (size_t i = 0; i < subscriptions_.size(); ++i)
    {
        auto sub = subscriptions_.copy_at(i);
        auto receiver_uid = to_cpu_uid(sub.get_receiver_uid());
        size_t type = sub.type();
        // TODO: do it in a proper way or rather just rework the whole mechanic.
        if (type == 0)
        {
            using MessageType = boost::mp11::mp_at_c<knp::core::messaging::MessageVariant, 0>;
            std::vector <knp::core::messaging::SpikeMessage> message_buf
                    = cpu_endpoint_.unload_messages<knp::core::messaging::SpikeMessage>(receiver_uid);
            for (auto &msg : message_buf)
            {
                send_message(make_gpu_message(msg));
            }
        }
        else if (type == 1)
        {
            using MessageType = boost::mp11::mp_at_c<knp::core::messaging::MessageVariant, 1>;
            std::vector<MessageType> message_buf
                    = cpu_endpoint_.unload_messages<MessageType>(receiver_uid);
            for (auto &msg : message_buf)
            {
                send_message(make_gpu_message(msg));
            }
        }
    }
}


namespace cm = knp::backends::gpu::cuda;

template
__host__ bool cm::CUDAMessageBus::subscribe_gpu<SpikeMessage>(const cm::UID&, const std::vector<cuda::UID>&);
template
__host__ bool cm::CUDAMessageBus::subscribe_gpu<SynapticImpactMessage>(const cm::UID&, const std::vector<cuda::UID>&);

template
__host__ bool cm::CUDAMessageBus::subscribe_both<SpikeMessage>(const cm::UID&, const std::vector<cuda::UID>&);
template
__host__ bool cm::CUDAMessageBus::subscribe_both<SynapticImpactMessage>(const cm::UID&, const std::vector<cuda::UID>&);


#define INSTANCE_MESSAGES_FUNCTIONS(n, template_for_instance, message_type)                \
    template bool CUDAMessageBus::unsubscribe<cm::message_type>(const cuda::UID &receiver);

BOOST_PP_SEQ_FOR_EACH(INSTANCE_MESSAGES_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_CUDA_MESSAGES))

}  // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpactMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
