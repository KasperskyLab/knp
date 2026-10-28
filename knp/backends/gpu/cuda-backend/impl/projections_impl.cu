/**
 * @file projections_impl.cu
 * @brief Contains functions for projection calculation.
 * @kaspersky_support A. Vartenkov
 * @date 14.08.2025
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

#include "projections_impl.cuh"
#include "cuda_lib/register_type.cuh"
#include "projection.cuh"
#include "resource_delta_projection.cuh"
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <string>
#include <vector>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
using ResourceSynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;

template<>
void gpu_insert<ProjectionVariants>(const ProjectionVariants &cpu_source, ProjectionVariants *gpu_target)
{
    ::cuda::std::visit([gpu_target](const auto &val)
    {
        using ValueType = std::decay_t<decltype(val)>;
        ValueType *buffer;
        call_and_check(cudaMalloc(&buffer, sizeof(ValueType)));
        gpu_insert(val, buffer);
        device_lib::make_variant_kernel<<<1, 1>>>(gpu_target, buffer);
        call_and_check(cudaFree(buffer));
    }, cpu_source);
}


__global__ void get_projection_uids_kernel(const ProjectionVariants *projection,
                                           cuda::UID *pre_uid,
                                           cuda::UID *post_uid,
                                           cuda::UID *self_uid)
{
    ::cuda::std::visit([pre_uid, post_uid, self_uid](const auto &proj)
    {
        *pre_uid = proj.presynaptic_uid_;
        *post_uid = proj.postsynaptic_uid_;
        *self_uid = proj.uid_;
    }, *projection);
}


auto get_projection_uids(const ProjectionVariants *proj)
{
    cuda::UID *pre_uid_gpu;
    cuda::UID *post_uid_gpu;
    cuda::UID *self_uid_gpu;
    cuda::UID pre_uid, post_uid, self_uid;
    call_and_check(cudaMalloc(&pre_uid_gpu, sizeof(cuda::UID)));
    call_and_check(cudaMalloc(&post_uid_gpu, sizeof(cuda::UID)));
    call_and_check(cudaMalloc(&self_uid_gpu, sizeof(cuda::UID)));
    get_projection_uids_kernel<<<1, 1>>>(proj, pre_uid_gpu, post_uid_gpu, self_uid_gpu);
    call_and_check(cudaMemcpy(&pre_uid, pre_uid_gpu, sizeof(cuda::UID), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&post_uid, post_uid_gpu, sizeof(cuda::UID), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&self_uid, self_uid_gpu, sizeof(cuda::UID), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(pre_uid_gpu));
    call_and_check(cudaFree(post_uid_gpu));
    call_and_check(cudaFree(self_uid_gpu));
    return std::make_tuple(pre_uid, post_uid, self_uid);
}


template <class DeltaLikeSynapse>
__global__ void calculate_synaptic_impact(
    const device_lib::CUDAVectorMutableView<typename CUDAProjection<DeltaLikeSynapse>::Synapse> synapses,
    const device_lib::LongIndex *synapse_indices, size_t size, StepIndex current_step, BitDelayBufferView buffer)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const device_lib::LongIndex synapse_id = synapse_indices[i];
    if (synapse_id >= synapses.size_)
    {
#ifndef NDEBUG
        printf("Warning: synapse id is beyond the bounds.");
#endif
        return;
    }
    auto delay = ::cuda::std::get<core::synapse_data>(synapses.data_[synapse_id]).delay_;
    // send_steps[i] = delay + current_step - 1;
    buffer.add_impact(synapse_id, delay, current_step);
}


template <>
__global__ void calculate_synaptic_impact<ResourceSynapseType>(
    const device_lib::CUDAVectorMutableView<typename CUDAProjection<ResourceSynapseType>::Synapse> synapses,
    const device_lib::LongIndex *synapse_indices, size_t size, StepIndex current_step, BitDelayBufferView buffer)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const device_lib::LongIndex synapse_id = synapse_indices[i];
    if (synapse_id >= synapses.size_) return; // TODO See if we need a warning
    auto delay = ::cuda::std::get<core::synapse_data>(synapses.data_[synapse_id]).delay_;
    buffer.add_impact(synapse_id, delay, current_step);
    ::cuda::std::get<core::synapse_data>(synapses.data_[synapse_id]).rule_.last_spike_step_ = current_step;
}

/**
 * @brief Calculate the series of impacts based on a single spike. Using index from projection we know the number of
 * synapses per each neuron, therefore we can know the output synapse index.
 * @param synapses all synapses, we use them for indexing.
 * @param spike_ids spike indexes, also the indexes of activated neurons
 * @param index synaptic index, see projection for more information
 * @param start_offsets output starting points for the synaptic impacts originating at the neuron.
 * @param current_step current step.
 * @param results synapse indexes, output.
 * @param send_steps sending steps, output.
 */
template <class DeltaLikeSynapse>
__global__ void calculate_impacts_per_spike(
    const device_lib::CUDAVectorMutableView<typename CUDAProjection<DeltaLikeSynapse>::Synapse> synapses,
    device_lib::CUDAVectorView<SpikeIndex> spike_ids, device_lib::IndexView index,
    device_lib::CUDAVectorView<device_lib::LongIndex> start_offsets, StepIndex current_step, BitDelayBufferView buffer)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Spike index.
    if (i >= spike_ids.size_ || i >= start_offsets.size_) return;
    const SpikeIndex neuron_id = spike_ids.data_[i];
    if (neuron_id + 1 >= index.offsets_size_) return;
    const device_lib::LongIndex start = index.offsets_ptr_[neuron_id];
    const device_lib::LongIndex size = index.offsets_ptr_[neuron_id + 1] - index.offsets_ptr_[neuron_id];
    const auto [num_blocks, num_threads] = device_lib::get_blocks_config(size);
    calculate_synaptic_impact<DeltaLikeSynapse><<<num_blocks, num_threads>>>(synapses, index.indices_ptr_ + start, size,
                                                                             current_step, buffer);
}


template <class DeltaLikeSynapse>
__global__ void buffered_spikes_to_impacts(
    device_lib::CUDAVectorView<typename CUDAProjection<DeltaLikeSynapse>::Synapse> synapses,
    BitDelayBufferView buffer, StepIndex current_step, device_lib::LongIndex *counter, SynapticImpact *impacts_out)
{
    const device_lib::LongIndex synapse_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_index >= synapses.size_) return;
    if (!buffer.has_impact(synapse_index, current_step)) return;
    auto &synapse = ::cuda::std::get<core::synapse_data>(synapses.data_[synapse_index]);

    SynapticImpact impact_out;
    impact_out.connection_index_ = synapse_index;
    impact_out.impact_value_ = synapse.weight_;
    impact_out.synapse_type_ = synapse.output_type_;
    impact_out.presynaptic_neuron_index_ = ::cuda::std::get<core::source_neuron_id>(synapses.data_[synapse_index]);
    impact_out.postsynaptic_neuron_index_ = ::cuda::std::get<core::target_neuron_id>(synapses.data_[synapse_index]);
    impacts_out[atomicAdd(counter, 1)] = impact_out;
}


template<>
void CUDAProjectionBase<knp::synapse_traits::DeltaSynapse>::form_message(StepIndex current_step)
{
    SPDLOG_DEBUG("Forming message for delta synapse");
    using Synapse = knp::synapse_traits::DeltaSynapse;

    SynapticImpact *impacts;
    device_lib::LongIndex *counter;
    // The simplest way, with a lot of memory overhead, but let's say we have as many impacts as there are synapses.
    call_and_check(cudaMalloc(&impacts, synapses_.size() * sizeof(SynapticImpact)));
    call_and_check(cudaMalloc(&counter, sizeof(device_lib::LongIndex)));
    call_and_check(cudaMemset(counter, 0, sizeof(device_lib::LongIndex)));
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapses_.size());
    buffered_spikes_to_impacts<Synapse><<<num_blocks, num_threads>>>(synapses_.view(), impact_buffer_.view(),
                                                                     current_step, counter, impacts);

    device_lib::LongIndex counter_cpu;
    cudaMemcpy(&counter_cpu, counter, sizeof(device_lib::LongIndex), cudaMemcpyDeviceToHost);
    if (counter_cpu == 0)
    {
        message_buf_.impacts_.clear();
        cudaFree(counter);
        cudaFree(impacts);
        return;
    }
    MessageHeader header{uid_, current_step};
    message_buf_.header_ = header;
    message_buf_.presynaptic_population_uid_ = presynaptic_uid_;
    message_buf_.postsynaptic_population_uid_ = postsynaptic_uid_;
    message_buf_.impacts_ = device_lib::CUDAVector<SynapticImpact>{impacts, counter_cpu};
    message_buf_.is_forcing_ = true;
    cudaFree(counter);
    SPDLOG_DEBUG("Formed message: step {}, num impacts {}, type {}", current_step, message_buf_.impacts_.size(),
           static_cast<int>(message_buf_.impacts_.copy_at(0).synapse_type_));
}


template<>
void CUDAProjectionBase<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>::form_message(StepIndex current_step)
{
    using Synapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;

    SynapticImpact *impacts;
    device_lib::LongIndex *counter;
    // The simplest way, with a lot of memory overhead, but let's say we have as many impacts as there are synapses.
    call_and_check(cudaMalloc(&impacts, synapses_.size() * sizeof(SynapticImpact)));
    call_and_check(cudaMalloc(&counter, sizeof(device_lib::LongIndex)));
    call_and_check(cudaMemset(counter, 0, sizeof(device_lib::LongIndex)));

    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapses_.size());
    buffered_spikes_to_impacts<Synapse><<<num_blocks, num_threads>>>(synapses_.view(), impact_buffer_.view(),
                                                                     current_step, counter, impacts);
    device_lib::LongIndex counter_cpu;
    cudaMemcpy(&counter_cpu, counter, sizeof(device_lib::LongIndex), cudaMemcpyDeviceToHost);
    if (counter_cpu == 0)
    {
        message_buf_.impacts_.clear();
        cudaFree(counter);
        cudaFree(impacts);
        return;
    }
    MessageHeader header{uid_, current_step};
    message_buf_.header_ = header;
    message_buf_.presynaptic_population_uid_ = presynaptic_uid_;
    message_buf_.postsynaptic_population_uid_ = postsynaptic_uid_;
    message_buf_.impacts_ = device_lib::CUDAVector<SynapticImpact>{impacts, counter_cpu};
    message_buf_.is_forcing_ = false;
}


/**
 * @brief Calculate delta projection, host gpu-using variant.
*/
__host__ void calculate_projection(
    CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection, const CUDAMessageBus &device_message_bus,
    const std::vector<device_lib::LongIndex> &message_ids, StepIndex step_n)
{
    using Synapse = knp::synapse_traits::DeltaSynapse;
    const auto &messages = device_message_bus.all_messages<SpikeMessage>();
    for (size_t i = 0; i < message_ids.size(); ++i)
    {
        const device_lib::LongIndex msg_index = message_ids[i];
        const size_t data_size = messages[msg_index].neuron_indexes_.size();
        auto msg_data_pointer_cpu = messages[msg_index].neuron_indexes_.data();
        SPDLOG_DEBUG("Got message data: pointer {}, size {}", reinterpret_cast<const void*>(msg_data_pointer_cpu),
                     data_size);

        if (data_size)
        {
            device_lib::LongIndex impacts_count = device_lib::count_values_by_indexes(projection.index_,
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});
            auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_size);
            auto output_start_indices = device_lib::calculate_neuron_scan(projection.index_,
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});

            calculate_impacts_per_spike<Synapse><<<num_blocks, num_threads>>>(projection.synapses_.mut_view(),
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size}, projection.index_.view(),
                    output_start_indices.view(), step_n, projection.impact_buffer_.view());

            cudaDeviceSynchronize();
        }
    }
    // Make messages
    projection.form_message(step_n);
}


// TODO: This functions are basically a copy of DeltaSynapse functions, we probably want to make them a template.
__host__ void calculate_projection(
    CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
    const CUDAMessageBus &device_message_bus, const std::vector<device_lib::LongIndex> &message_ids, StepIndex step_n)
{
    using Synapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    SPDLOG_DEBUG("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(to_cpu_uid(projection.uid_)));
    const auto &messages = device_message_bus.all_messages<SpikeMessage>();
    for (size_t i = 0; i < message_ids.size(); ++i)
    {
        const device_lib::LongIndex msg_index = message_ids[i];
        const size_t data_size = messages[msg_index].neuron_indexes_.size();
        auto msg_data_pointer_cpu = messages[msg_index].neuron_indexes_.data();
        SPDLOG_TRACE("Got message data: pointer {}, size {}", reinterpret_cast<const void*>(msg_data_pointer_cpu),
                     data_size);

        if (data_size)
        {
            auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_size);
            auto output_start_indices = device_lib::calculate_neuron_scan(
                    projection.index_,
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});

            calculate_impacts_per_spike<Synapse><<<num_blocks, num_threads>>>(projection.synapses_.mut_view(),
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size}, projection.index_.view(),
                    output_start_indices.view(), step_n, projection.impact_buffer_.view());

            cudaDeviceSynchronize();
        }
    }
    // Make synaptic impact message from the impacts calculated above and store it inside an internal buffer.
    projection.form_message(step_n);
}


__host__ void calculate_projection(
    CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
    const std::vector<device_lib::LongIndex> &message_ids,
    StepIndex step_n)
{
    SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}, not implemented.",
                 std::string(to_cpu_uid(projection.uid_)));
}
} // namespace knp::backends::gpu::cuda

using ResourceSynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceSynapseValue = typename knp::backends::gpu::cuda::CUDAProjection<ResourceSynapseType>::Synapse;
using ResourceSynapseValuePtr = ResourceSynapseValue*;

REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(ResourceSynapseValue);
REGISTER_CUDA_VECTOR_TYPE(ResourceSynapseValuePtr);
