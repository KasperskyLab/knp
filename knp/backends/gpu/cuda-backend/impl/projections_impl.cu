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
    if (synapse_id >= synapses.size_) return; // TODO See if we need a warning
    auto delay = ::cuda::std::get<0>(synapses.data_[synapse_id]).delay_;
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
    auto delay = ::cuda::std::get<0>(synapses.data_[synapse_id]).delay_;
    buffer.add_impact(synapse_id, delay, current_step);
    ::cuda::std::get<0>(synapses.data_[synapse_id]).rule_.last_spike_step_ = current_step;

}

/**
 * @brief Calculate the series of impacts based on a single spike. Using index from projection we know the number of
 * synapses per each neuron, therefore we can know the output synapse index.
 * @param synapses all synapses, we use them for indexing.
 * @param spike_ids spike indexes, also the indexes of activated neurons
 * @param index synaptic index, see projection for more information
 * @param start_offsets output starting points for the synaptic impacts originating at the neuron.
 * @param current_step current step
 * @param results synapse indexes, output
 * @param send_steps sending steps, output
 */
template <class DeltaLikeSynapse>
__global__ void calculate_impacts_per_spike(
        const device_lib::CUDAVectorMutableView<typename CUDAProjection<DeltaLikeSynapse>::Synapse> synapses,
        device_lib::CUDAVectorView<SpikeIndex> spike_ids, device_lib::IndexView index,
        device_lib::CUDAVectorView<device_lib::LongIndex> start_offsets,
        StepIndex current_step, BitDelayBufferView buffer)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Spike index.
    if (i >= spike_ids.size_ || i >= start_offsets.size_) return;
    const SpikeIndex neuron_id = spike_ids.data_[i];
    if (neuron_id + 1 >= index.offsets_size_) return;
    const device_lib::LongIndex start = index.offsets_ptr_[neuron_id];
    const device_lib::LongIndex size = index.offsets_ptr_[neuron_id + 1] - index.offsets_ptr_[neuron_id];
    const device_lib::LongIndex output_start = start_offsets.data_[i];
    const auto [num_blocks, num_threads] = device_lib::get_blocks_config(size);
    // printf("Calc impacts: start %lu, size %lu, out_start %lu\n", start, size, output_start);
    calculate_synaptic_impact<DeltaLikeSynapse><<<num_blocks, num_threads>>>(synapses, index.indices_ptr_ + start, size,
            current_step, buffer);
}


//template <class DeltaLikeSynapse>
//__global__ void delta_indices_to_impacts_kernel(device_lib::LongIndex *indices_begin,
//        device_lib::LongIndex *indices_end,
//        cuda::device_lib::CUDAVectorView<typename CUDAProjection<DeltaLikeSynapse>::Synapse> synapses,
//        SynapticImpact *impacts_out)
//{
//    constexpr int data_index = core::SynapseElementAccess::synapse_data;
//    constexpr int source_id_index = core::SynapseElementAccess::source_neuron_id;
//    constexpr int target_id_index = core::SynapseElementAccess::target_neuron_id;
//
//    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//    device_lib::LongIndex *index = indices_begin + i;
//    if (index >= indices_end) return;
//    const device_lib::LongIndex synapse_id = *index;
//    if (synapse_id >= synapses.size_) return;
//    auto &synapse = ::cuda::std::get<data_index>(synapses.data_[synapse_id]);
//    SynapticImpact impact_out;
//    impact_out.connection_index_ = synapse_id;
//    impact_out.impact_value_ = synapse.weight_;
//    impact_out.synapse_type_ = synapse.output_type_;
//    impact_out.presynaptic_neuron_index_ = ::cuda::std::get<source_id_index>(synapses.data_[synapse_id]);
//    impact_out.postsynaptic_neuron_index_ = ::cuda::std::get<target_id_index>(synapses.data_[synapse_id]);
//    impacts_out[i] = impact_out;
//}


template <class DeltaLikeSynapse>
__global__ void buffered_spikes_to_impacts(
        device_lib::CUDAVectorView<typename CUDAProjection<DeltaLikeSynapse>::Synapse> synapses,
        BitDelayBufferView buffer, StepIndex current_step, device_lib::LongIndex *counter, SynapticImpact *impacts_out)
{
    const device_lib::LongIndex synapse_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_) return;
    if (!buffer.has_impact(synapse_index, current_step)) return;
    auto &synapse = ::cuda::std::get<data_index>(synapses.data_[synapse_id]);

    SynapticImpact impact_out;
    impact_out.connection_index_ = synapse_id;
    impact_out.impact_value_ = synapse.weight_;
    impact_out.synapse_type_ = synapse.output_type_;
    impact_out.presynaptic_neuron_index_ = ::cuda::std::get<source_id_index>(synapses.data_[synapse_id]);
    impact_out.postsynaptic_neuron_index_ = ::cuda::std::get<target_id_index>(synapses.data_[synapse_id]);
    impacts_out[atomicAdd(counter, 1)] = impact_out;
}


template<>
void CUDAProjectionBase<knp::synapse_traits::DeltaSynapse>::form_message(StepIndex current_step)
{
    SPDLOG_DEBUG("Forming message for delta synapse");
    using Synapse = knp::synapse_traits::DeltaSynapse;
//    auto iter = thrust::upper_bound(thrust::device, sending_steps_.begin(), sending_steps_.end(), current_step);
//    if (iter == sending_steps_.begin())
//    {
//        // message_buf_.impacts_.clear();
//        return;
//    }
//    device_lib::LongIndex num_impacts = iter - sending_steps_.begin();
//
//    SynapticImpact *impacts;
//    auto [num_blocks, num_threads] = device_lib::get_blocks_config(num_impacts);
//    call_and_check(cudaMalloc(&impacts, sizeof(SynapticImpact) * num_impacts));
//    delta_indices_to_impacts_kernel<Synapse><<<num_blocks, num_threads>>>(impact_indexes_.data(),
//            impact_indexes_.data() + num_impacts, synapses_.view(), impacts);

    SynapticImpact *impacts;
    device_lib::LongIndex *counter;
    // The simplest way, with a lot of memory overhead, but let's say we have as many impacts as there are synapses.
    device_lib::call_and_check(cudaMalloc(&impacts, synapses_.size() * sizeof(SynapticImpact)));
    device_lib::call_and_check(cudaMalloc(&counter, sizeof(device_lib::LongIndex)));
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapses_.size());
    buffered_spikes_to_impacts<<<num_blocks, num_threads>>>(synapses_.view(), impact_buffer_.view(), step, counter,
                                                            impacts);
    MessageHeader header{uid_, current_step};
    message_buf_.header_ = header;
    message_buf_.presynaptic_population_uid_ = presynaptic_uid_;
    message_buf_.postsynaptic_population_uid_ = postsynaptic_uid_;
    message_buf_.impacts_ = device_lib::CUDAVector<SynapticImpact>{impacts, num_impacts};
    message_buf_.is_forcing_ = true;
    cudaFree(counter);
    printf("Formed message: step %llu, num impacts %llu, type %d\n", current_step, message_buf_.impacts_.size(),
           static_cast<int>(message_buf_.impacts_.copy_at(0).synapse_type_));
}


template<>
void CUDAProjectionBase<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>::form_message(StepIndex current_step)
{
//    using Synapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
//    auto iter = thrust::upper_bound(thrust::device, sending_steps_.begin(), sending_steps_.end(), current_step);
//    if (iter == sending_steps_.begin())
//    {
//        // message_buf_.impacts_.clear();
//        return;
//    }
//    device_lib::LongIndex num_impacts = iter - sending_steps_.begin();
//
//    SynapticImpact *impacts;
//    auto [num_blocks, num_threads] = device_lib::get_blocks_config(num_impacts);
//    call_and_check(cudaMalloc(&impacts, sizeof(SynapticImpact) * num_impacts));
//    delta_indices_to_impacts_kernel<Synapse><<<num_blocks, num_threads>>>(impact_indexes_.data(),
//            impact_indexes_.data() + num_impacts, synapses_.view(), impacts);
//    MessageHeader header{uid_, current_step};
//    message_buf_.header_ = header;
//    message_buf_.presynaptic_population_uid_ = presynaptic_uid_;
//    message_buf_.postsynaptic_population_uid_ = postsynaptic_uid_;
//    message_buf_.impacts_ = device_lib::CUDAVector<SynapticImpact>{impacts, num_impacts};
    using Synapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;

    SynapticImpact *impacts;
    device_lib::LongIndex *counter;
    // The simplest way, with a lot of memory overhead, but let's say we have as many impacts as there are synapses.
    device_lib::call_and_check(cudaMalloc(&impacts, synapses_.size() * sizeof(SynapticImpact)));
    device_lib::call_and_check(cudaMalloc(&counter, sizeof(device_lib::LongIndex)));
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapses_.size());
    buffered_spikes_to_impacts<<<num_blocks, num_threads>>>(synapses_.view(), impact_buffer_.view(), step, counter,
                                                            impacts);
    MessageHeader header{uid_, current_step};
    message_buf_.header_ = header;
    message_buf_.presynaptic_population_uid_ = presynaptic_uid_;
    message_buf_.postsynaptic_population_uid_ = postsynaptic_uid_;
    message_buf_.impacts_ = device_lib::CUDAVector<SynapticImpact>{impacts, num_impacts};
    message_buf_.is_forcing_ = false;
    sending_steps_.erase(sending_steps_.begin(), sending_steps_.begin() + num_impacts);
    impact_indexes_.erase(impact_indexes_.begin(), impact_indexes_.begin() + num_impacts);
}


/**
 * @brief Calculate delta projection, host gpu-using variant.
*/
__host__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection,
        const CUDAMessageBus &device_message_bus,
        const std::vector<device_lib::LongIndex> &message_ids,
        StepIndex step_n)
{
    using Synapse = knp::synapse_traits::DeltaSynapse;
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
            device_lib::LongIndex impacts_count = count_values_by_indexes(projection.index_,
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});
            printf("Impacts count: %llu\n", impacts_count);

//            device_lib::LongIndex *impacts_buffer;
//            device_lib::LongIndex *delay_buffer;
//            call_and_check(cudaMalloc(&impacts_buffer, sizeof(device_lib::LongIndex) * impacts_count));
//            call_and_check(cudaMalloc(&delay_buffer, sizeof(device_lib::LongIndex) * impacts_count));

            // 2. For each active synapse calculate its impact.
            auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_size);
            auto output_start_indices = device_lib::calculate_neuron_scan(projection.index_,
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});

            calculate_impacts_per_spike<Synapse><<<num_blocks, num_threads>>>(projection.synapses_.mut_view(),
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size}, projection.index_.view(),
                    output_start_indices.view(), step_n, impact_buffer_.view());

            cudaDeviceSynchronize();
//            // 3. Sort impacts by time
//            thrust::sort_by_key(thrust::device, delay_buffer, delay_buffer + impacts_count, impacts_buffer);
//            projection.add_impacts(device_lib::CUDAVector<device_lib::LongIndex>{impacts_buffer, impacts_count},
//                                   device_lib::CUDAVector<device_lib::LongIndex>{delay_buffer, impacts_count});
            // TODO TEMP
            {
                const auto synapse = ::cuda::std::get<0>(projection.synapses_.copy_at(0));
                if (static_cast<int>(synapse.output_type_) == 3)
                    printf("Projection of size %llu got message of size %lu; impacts %llu\n",
                           projection.synapses_.size(), data_size, impacts_count);
            }
            // END TEMP
        }
    }
    // Make messages
    projection.form_message(step_n);
}


__host__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
        const CUDAMessageBus &device_message_bus,
        const std::vector<device_lib::LongIndex> &message_ids,
        StepIndex step_n)
{
    using Synapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
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
//            device_lib::LongIndex impacts_count = count_values_by_indexes(projection.index_,
//                                                                          device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});

//            device_lib::LongIndex *impacts_buffer;
//            device_lib::LongIndex *delay_buffer;
//            call_and_check(cudaMalloc(&impacts_buffer, sizeof(device_lib::LongIndex) * impacts_count));
//            call_and_check(cudaMalloc(&delay_buffer, sizeof(device_lib::LongIndex) * impacts_count));

            // 2. For each active synapse calculate its impact.
            auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_size);
            auto output_start_indices = device_lib::calculate_neuron_scan(
                    projection.index_,
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size});

            calculate_impacts_per_spike<Synapse><<<num_blocks, num_threads>>>(projection.synapses_.mut_view(),
                    device_lib::CUDAVectorView<SpikeIndex>{msg_data_pointer_cpu, data_size}, projection.index_.view(),
                    output_start_indices.view(), step_n, impact_buffer_.view());

            cudaDeviceSynchronize();
//            // 3. Sort impacts by time
//            thrust::sort_by_key(thrust::device, delay_buffer, delay_buffer + impacts_count, impacts_buffer);
//            projection.add_impacts(device_lib::CUDAVector<device_lib::LongIndex>{impacts_buffer, impacts_count},
//                                   device_lib::CUDAVector<device_lib::LongIndex>{delay_buffer, impacts_count});
        }
    }
    // Make messages
    projection.form_message(step_n);
}


// TODO: This functions are basically a copy of DeltaSynapse functions, we probably want to make them a template.


__host__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
        const std::vector<device_lib::LongIndex> &message_ids,
        StepIndex step_n)
{
    //SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
}
} // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
