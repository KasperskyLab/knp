/**
 * @file messaging.cu
 * @brief Messages file for CUDA.
 * @kaspersky_support Artiom N.
 * @date 26.09.2025
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

#include "messaging.cuh"
#include "../cuda_lib/kernels.cuh"
#include "subscription.cuh"
#include "../cuda_lib/extraction.cuh"
#include "../cuda_lib/register_all.cuh"
#include "../cuda_lib/vector_kernels.cuh"

#include <type_traits>

/**
 * @brief CUDA messaging namespace.
 */
namespace knp::backends::gpu::cuda
{
template<size_t index>
MessageVariant extract_message_by_index(const void *msg_ptr)
{
    return gpu_extract<boost::mp11::mp_at_c<AllCudaMessages, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<AllCudaMessages, index> *>(msg_ptr));
}


template<size_t Index>
inline void extract_dispatch(MessageVariant &result, size_t type, const void *msg_ptr)
{
    if (Index - 1 == type)
    {
        result = extract_message_by_index<Index - 1>(msg_ptr);
        return;
    }
    if constexpr (Index == 1)
    {
        throw std::runtime_error("Wrong message type index when extracting");
    }
    else
    {
        extract_dispatch<Index - 1>(result, type, msg_ptr);
    }
}


template<>
MessageVariant gpu_extract<MessageVariant>(const MessageVariant *message)
{
    int *type_gpu;
    // This is a gpu pointer to gpu pointer to gpu message.
    const void **msg_gpu;
    call_and_check(cudaMalloc(&type_gpu, sizeof(int)));
    call_and_check(cudaMalloc(&msg_gpu, sizeof(void *)));
    get_message_kernel<<<1, 1>>>(message, type_gpu, msg_gpu);
    int type;

    // This is a gpu pointer to gpu message. &msg_ptr is a cpu pointer to gpu pointer to gpu message.
    const void *msg_ptr;
    call_and_check(cudaMemcpy(&type, type_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&msg_ptr, msg_gpu, sizeof(void *), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(type_gpu));
    call_and_check(cudaFree(msg_gpu));
    // Here we have a type index and a gpu pointer to message.
    MessageVariant result;
    extract_dispatch<::cuda::std::variant_size_v<cuda::MessageVariant>>(result, type, msg_ptr);
    return result;
}


template<>
void gpu_insert<MessageVariant>(const MessageVariant &cpu_source, MessageVariant *gpu_target)
{
    ::cuda::std::visit([gpu_target](const auto &val)
    {
        using ValueType = std::decay_t<decltype(val)>;
        ValueType *buffer;
        call_and_check(cudaMalloc(&buffer, sizeof(ValueType)));
        gpu_insert(val, buffer);
        device_lib::make_variant_kernel<<<1, 1>>>(gpu_target, buffer);
        device_lib::destruct_kernel<ValueType, device_lib::CuMallocAllocator<ValueType>><<<1, 1>>>(buffer, 1);
        call_and_check(cudaFree(buffer));
    }, cpu_source);
}


cuda::SpikeMessage make_gpu_message(const knp::core::messaging::SpikeMessage &host_message)
{
    cuda::SpikeMessage result;
    result.header_ = detail::make_gpu_message_header(host_message.header_);
    size_t data_n = host_message.neuron_indexes_.size();
    static_assert(std::is_same_v<cuda::SpikeIndex, knp::core::messaging::SpikeIndex>);
    if (data_n != 0)
    {
        result.neuron_indexes_.resize(data_n);
        call_and_check(cudaMemcpy(result.neuron_indexes_.data(), host_message.neuron_indexes_.data(),
                                  data_n * sizeof(cuda::SpikeIndex), cudaMemcpyHostToDevice));
    }
    return result;
}


knp::core::messaging::SpikeMessage make_host_message(const cuda::SpikeMessage &gpu_message)
{
    knp::core::messaging::SpikeMessage result;
    result.header_ = detail::make_host_message_header(gpu_message.header_);
    size_t data_n = gpu_message.neuron_indexes_.size();
    static_assert(std::is_same_v<cuda::SpikeIndex, knp::core::messaging::SpikeIndex>);
    if (data_n != 0)
    {
        result.neuron_indexes_.resize(data_n);
        call_and_check(cudaMemcpy(result.neuron_indexes_.data(), gpu_message.neuron_indexes_.data(),
                                  data_n * sizeof(cuda::SpikeIndex), cudaMemcpyDeviceToHost));
    }
    return result;
}



__global__ void copy_host_impact_kernel(cuda::SynapticImpact *impacts_to,
                                        const knp::core::messaging::SynapticImpact *impacts_from,
                                        size_t num_impacts)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_impacts) return;
    *(impacts_to + i) = detail::make_gpu_impact(*(impacts_from + i));
}


__global__ void copy_gpu_impact_kernel(knp::core::messaging::SynapticImpact *impacts_to,
                                       const cuda::SynapticImpact *impacts_from,
                                       size_t num_impacts)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_impacts) return;
    *(impacts_to + i) = detail::make_host_impact(*(impacts_from + i));
}


cuda::SynapticImpactMessage make_gpu_message(const knp::core::messaging::SynapticImpactMessage &host_message)
{
    cuda::SynapticImpactMessage result;
    // Copy necessary fields.
    result.header_ = detail::make_gpu_message_header(host_message.header_);
    result.presynaptic_population_uid_ = to_gpu_uid(host_message.presynaptic_population_uid_);
    result.postsynaptic_population_uid_ = to_gpu_uid(host_message.postsynaptic_population_uid_);
    result.is_forcing_ = host_message.is_forcing_;
    size_t data_n = host_message.impacts_.size();

    if (!data_n) return result;
    // Copy data if it exists.
    size_t data_size = data_n * sizeof(knp::core::messaging::SynapticImpact);
    knp::core::messaging::SynapticImpact *in_data;
    call_and_check(cudaMalloc(&in_data, data_size));
    call_and_check(cudaMemcpy(in_data, host_message.impacts_.data(), data_size, cudaMemcpyHostToDevice));
    result.impacts_.resize(data_n);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_n);
    copy_host_impact_kernel<<<num_blocks, num_threads>>>(result.impacts_.data(), in_data, data_n);
    call_and_check(cudaFree(in_data));
    return result;
}


knp::core::messaging::SynapticImpactMessage make_host_message(const cuda::SynapticImpactMessage &gpu_message)
{
    knp::core::messaging::SynapticImpactMessage result;
    // Copy necessary fields.
    result.header_ = detail::make_host_message_header(gpu_message.header_);
    result.presynaptic_population_uid_ = to_cpu_uid(gpu_message.presynaptic_population_uid_);
    result.postsynaptic_population_uid_ = to_cpu_uid(gpu_message.postsynaptic_population_uid_);
    result.is_forcing_ = gpu_message.is_forcing_;
    size_t data_n = gpu_message.impacts_.size();

    if (!data_n) return result;

    // Copy data if it exists.
    size_t data_size = data_n * sizeof(knp::core::messaging::SynapticImpact);
    knp::core::messaging::SynapticImpact *in_data;
    call_and_check(cudaMalloc(&in_data, data_size));

    result.impacts_.resize(data_n);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_n);
    copy_gpu_impact_kernel<<<num_blocks, num_threads>>>(in_data, gpu_message.impacts_.data(), data_n);

    result.impacts_.resize(data_n);
    call_and_check(cudaMemcpy(result.impacts_.data(), in_data, data_size, cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(in_data));
    return result;
}


knp::core::messaging::MessageVariant make_host_message(const cuda::MessageVariant &gpu_message)
{
    return ::cuda::std::visit([](const auto &msg)
    {
        return knp::core::messaging::MessageVariant{make_host_message(msg)};
    }, gpu_message);
}


cuda::MessageVariant make_gpu_message(const knp::core::messaging::MessageVariant &host_message)
{
    return std::visit([](const auto &msg)
    {
        return cuda::MessageVariant{make_gpu_message(msg)};
    }, host_message);
}

} // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpactMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
