/**
 * @file resource_blifat_population_impl.cu
 * @brief Contains functions for Resource BLIFAT population calculation.
 * @kaspersky_support A. Vartenkov
 * @date 17.08.2026
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

#include "resource_stdp_impl.cuh"
#include "../cuda_lib/vector.cuh"


namespace knp::backends::gpu::cuda
{
__device__ void update_isi(ResourceBlifatParams &neuron, StepIndex step)
{
    // This neuron got a forcing spike this turn and doesn't continue its spiking sequence.
    if (neuron.is_being_forced_)
    {
        neuron.isi_status_ = neuron_traits::ISIPeriodType::is_forced;
        return;
    }

    switch (neuron.isi_status_)
    {
        case neuron_traits::ISIPeriodType::not_in_period:
        case neuron_traits::ISIPeriodType::is_forced:
            neuron.isi_status_ = neuron_traits::ISIPeriodType::period_started;
            neuron.first_isi_spike_ = step;
            break;
        case neuron_traits::ISIPeriodType::period_started:
            if (step - neuron.last_step_ < neuron.isi_max_)
            {
                neuron.isi_status_ = neuron_traits::ISIPeriodType::period_continued;
            }
            break;
        case neuron_traits::ISIPeriodType::period_continued:
            if (step - neuron.last_step_ >= neuron.isi_max_ || neuron.dopamine_value_ != 0)
            {
                neuron.isi_status_ = neuron_traits::ISIPeriodType::period_started;
                neuron.first_isi_spike_ = step;
            }
            break;
        default:
            printf("Error: unsupported neuron state\n");
    }

    neuron.last_step_ = step;
}


template <class Value>
__host__ __device__ bool is_point_in_interval(Value interval_begin, Value interval_end, Value point)
{
    if (interval_begin > interval_end) ::cuda::std::swap(interval_begin, interval_end);
    return (interval_begin <= point) && (point <= interval_end);
}


__global__ void process_synapses_per_neuron_kernel(ResourceBlifatParams *neuron,
                                                   device_lib::CUDAVectorMutableView<SynapseValue*> synapses,
                                                   size_t neuron_index,
                                                   StepIndex step)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_) return;
    ResourceSynapseParams &synapse = ::cuda::std::get<core::synapse_data>(*synapses.data_[synapse_id]);
    float threshold_update = synapse.weight_ * (synapse.weight_ > 0) * neuron->synapse_sum_threshold_coefficient_;
    // Technically a race condition, float operations are not order-independent, but oh well.
    atomicAdd(&neuron->additional_threshold_, threshold_update);

    const bool had_spike = is_point_in_interval<StepIndex>(step - synapse.rule_.dopamine_plasticity_period_,
                                                           step,
                                                           synapse.rule_.last_spike_step_ + synapse.delay_ - 1);
    // While period continues we don't change has_contributed from true to false.
    if (neuron_traits::ISIPeriodType::period_continued != neuron->isi_status_ || had_spike)
        synapse.rule_.has_contributed_ = had_spike;

    if (neuron->isi_status_ != neuron_traits::ISIPeriodType::period_continued)
        synapse.rule_.had_hebbian_update_ = false;

    if (neuron->isi_status_ != neuron_traits::ISIPeriodType::is_forced)
    {
        // Unconditional decreasing synaptic resource.
        // TODO: NOT HERE. This shouldn't matter now as d_u_ is zero for our task, but the logic should be changed.
        synapse.rule_.synaptic_resource_ -= synapse.rule_.d_u_;
        atomicAdd(&neuron->free_synaptic_resource_, synapse.rule_.d_u_);
        // Hebbian plasticity.
        // Check if synapse ever got a spike in the current ISI period.
        if (synapse.rule_.has_contributed_ && !synapse.rule_.had_hebbian_update_)
        {
            // 2. If it did, then update synaptic resource value.
            const float d_h = neuron->d_h_ * ::cuda::std::min(static_cast<float>(
                ::cuda::std::pow(2, -neuron->stability_)), 1.F);
            atomicAdd(&synapse.rule_.synaptic_resource_, d_h);
            atomicAdd(&neuron->free_synaptic_resource_, -d_h);
            synapse.rule_.had_hebbian_update_ = true;
        }
    }
    // Recalculating synapse weights. Sometimes it probably doesn't need to happen, check it later.
    recalculate_synapse_weight(synapse);
}


__global__ void process_spiking_neurons_kernel(device_lib::CUDAVectorView<SpikeIndex> spikes,
                                               device_lib::CUDAVectorMutableView<ResourceBlifatParams> neurons,
                                               SynapsesPerNeuronsView synapses,
                                               StepIndex step)
{
    const device_lib::LongIndex spike_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (spike_id >= spikes.size_) return;
    const auto neuron_id = spikes.data_[spike_id];
    ResourceBlifatParams &neuron = neurons.data_[neuron_id];
    neuron.last_spike_step_ = step;
    // Calculate neuron ISI status.
    update_isi(neuron, step);
    if (knp::neuron_traits::ISIPeriodType::period_started == neuron.isi_status_)
        neuron.stability_ -= neuron.stability_change_at_isi_;
    neuron.additional_threshold_ = 0.0;
    device_lib::CUDAVectorMutableView<SynapseValue*> synapse_view = extract_synapses_from_index(synapses, neuron_id);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapse_view.size_);
    process_synapses_per_neuron_kernel<<<num_blocks, num_threads>>>(&neuron, synapse_view, neuron_id, step);
}

void process_spiking_neurons_impl(const device_lib::CUDAVector<SpikeIndex> &msg, SynapsesPerNeuronsView &synapse_index,
                                  device_lib::CUDAVector<ResourceBlifatParams> &neurons, StepIndex step)
{
    // It's very important that during this function no projection invalidates iterators.
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.size());
    process_spiking_neurons_kernel<<<num_blocks, num_threads>>>(msg.view(), neurons.mut_view(), synapse_index, step);
}


__device__ void recalculate_synapse_weight(ResourceSynapseParams &synapse)
{
    const auto syn_w = std::max(synapse.rule_.synaptic_resource_, 0.F);
    const auto weight_diff = synapse.rule_.w_max_ - synapse.rule_.w_min_;
#if (!defined(NDEBUG) && defined(FULL_DEBUG))
    const auto weight_before = synapse.weight_;
#endif // !NDEBUG && FULL_DEBUG
    synapse.weight_ = synapse.rule_.w_min_ + weight_diff * syn_w / (weight_diff + syn_w);
#if (!defined(NDEBUG) && defined(FULL_DEBUG))
    if (abs(weight_before - synapse.weight_) >= 1e-5)
        printf("Weight before %f, after %f\n", weight_before, synapse.weight_);
#endif // !NDEBUG && FULL_DEBUG
}


__global__ void index_to_pointer(device_lib::IndexView synapse_index, SynapseValue *start, SynapseValue **output)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapse_index.indices_size_) return;
    output[synapse_id] = start + synapse_index.indices_ptr_[synapse_id];
}


__host__ SynapsesPerNeurons initialize_synapses_per_neurons(
        const device_lib::IndexView &synapse_index,
        SynapseValue *start)
{
    SynapsesPerNeurons result;
    SPDLOG_DEBUG("Initializing synapses_per_neurons.");
    result.offsets_.resize(synapse_index.offsets_size_);
    result.synapses_.resize(synapse_index.indices_size_);
    cudaMemcpy(result.offsets_.data(), synapse_index.offsets_ptr_,
               sizeof(device_lib::LongIndex) * synapse_index.offsets_size_, cudaMemcpyDeviceToDevice);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapse_index.indices_size_);
    index_to_pointer<<<num_blocks, num_threads>>>(synapse_index, start, result.synapses_.data());
    cudaDeviceSynchronize();
    return result;
}
// TODO: Merge synapses pointers


__global__ void add_resource_to_synapses(device_lib::CUDAVectorMutableView<SynapseValue*> synapses,
                                         double add_resource_value)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_) return;
    auto &synapse = ::cuda::std::get<core::synapse_data>(*synapses.data_[synapse_id]);
#ifndef NDEBUG
    printf("Adding to synapse resource %f value %f\n", synapse.rule_.synaptic_resource_, add_resource_value);
#endif // NDEBUG
    synapse.rule_.synaptic_resource_ += add_resource_value;
    recalculate_synapse_weight(synapse);
}


__device__ void renormalize_resource(device_lib::CUDAVectorMutableView<SynapseValue*> synapses,
                                     ResourceBlifatParams &neuron, StepIndex step)
{
    if (step - neuron.last_step_ <= neuron.isi_max_ &&
        neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced)
    {
        // Neuron is still in ISI period, skip it.
        return;
    }

    if (::cuda::std::fabs(neuron.free_synaptic_resource_) < neuron.synaptic_resource_threshold_)
    {
        // Not enough resource for the threshold, skip renormalizing.
        return;
    }

    // Divide free resource between all synapses.
    auto add_resource_value =
            neuron.free_synaptic_resource_ / (synapses.size_ + neuron.resource_drain_coefficient_);

    neuron.free_synaptic_resource_ = 0.0F;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapses.size_);
    add_resource_to_synapses<<<num_blocks, num_threads>>>(synapses, add_resource_value);
}


template <class Synapse>
__global__ void do_dopamine_plasticity_synapse_kernel(
        device_lib::CUDAVectorMutableView<SynapseValue*> synapses,
        ResourceBlifatParams *neuron, StepIndex step)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_ || neuron->dopamine_value_ == 0) return;
    auto &synapse = ::cuda::std::get<core::synapse_data>(*synapses.data_[synapse_id]);
    if (step - neuron->last_spike_step_ <= neuron->dopamine_plasticity_time_ && synapse.rule_.has_contributed_)
    {
        // Change synapse resource.
        float resource_change =
            neuron->dopamine_value_ * ::cuda::std::min(static_cast<float>(std::pow(2, -neuron->stability_)), 1.F);
        synapse.rule_.synaptic_resource_ += resource_change;
        atomicAdd(&neuron->free_synaptic_resource_, -resource_change);
    }
    recalculate_synapse_weight(synapse);
}


template<class Synapse>
__device__ void do_dopamine_plasticity_device(
        device_lib::CUDAVectorMutableView<SynapseValue*> synapses,
        ResourceBlifatParams &neuron, StepIndex step)
{
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapses.size_);

    if (neuron.dopamine_value_ <= 0 && neuron.isi_status_ == neuron_traits::ISIPeriodType::is_forced) return;

    do_dopamine_plasticity_synapse_kernel<ResourceSynapseType><<<num_blocks, num_threads>>>(synapses, &neuron, step);
}


__device__ void update_neuron_stability(ResourceBlifatParams &neuron, StepIndex step)
{
    if (neuron.dopamine_value_ == 0) return;
    if (neuron.isi_status_ == neuron_traits::ISIPeriodType::is_forced || neuron.dopamine_value_ < 0)
    {
        // A dopamine reward when forced or a dopamine punishment reduce stability by `r * D`.
        neuron.stability_ -= neuron.dopamine_value_ * neuron.stability_change_parameter_;
        neuron.stability_ = std::max(neuron.stability_, 0.0F);
    }
    else
    {
        // A dopamine reward when non-forced changes stability by `D max(2 - |t(TSS) - ISImax| / ISImax, -1)`.
        const double dopamine_constant = 2.0;
        const double difference = std::fabs(step - neuron.first_isi_spike_ - neuron.isi_max_);
        neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
            std::max(dopamine_constant - difference / neuron.isi_max_, -1.0);
    }
}


__global__ void do_dopamine_plasticity_kernel(SynapsesPerNeuronsView synapse_pointer_index,
                                              device_lib::CUDAVectorMutableView<ResourceBlifatParams> neurons,
                                              StepIndex step)
{
    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    using SynapseParamType = knp::synapse_traits::synapse_parameters<SynapseType>;
    // Find the current neuron.
    const device_lib::LongIndex neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Check that it's a correct kernel.
    if (neuron_id >= neurons.size_) return;
    // Check that it's not an unconnected "extra" neuron.
    if (synapse_pointer_index.offsets_size_ == 0 || neuron_id >= synapse_pointer_index.offsets_size_ - 1) return;
    if (neurons.data_[neuron_id].dopamine_value_ != 0)
    {
        do_dopamine_plasticity_device<ResourceSynapseType>(
            extract_synapses_from_index(synapse_pointer_index, neuron_id),
            neurons.data_[neuron_id], step);
    }
}


__global__ void update_and_renormalize(SynapsesPerNeuronsView synapse_pointer_index,
                                       device_lib::CUDAVectorMutableView<ResourceBlifatParams> neurons, StepIndex step)
{
    const device_lib::LongIndex neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= neurons.size_) return;
    update_neuron_stability(neurons.data_[neuron_id], step);
    if (synapse_pointer_index.offsets_size_ == 0 || neuron_id >= synapse_pointer_index.offsets_size_ - 1) return;
    renormalize_resource(extract_synapses_from_index(synapse_pointer_index, neuron_id), neurons.data_[neuron_id], step);
}


void update_rstdp_synapses(CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
                           SynapsesPerNeuronsView synapses, const device_lib::CUDAVector<SpikeIndex> &spiking_neurons,
                           StepIndex step)
{
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(population.neurons_.size());
    if (spiking_neurons.size())
    {
#ifndef NDEBUG
        printf("Spiked neurons ");
        for (size_t i = 0; i < spiking_neurons.size(); ++i)
            printf("%u ", spiking_neurons.copy_at(i));
        printf("\n");
#endif // NDEBUG
        process_spiking_neurons_impl(spiking_neurons, synapses, population.neurons_, step);
    }
    cudaDeviceSynchronize();
    SPDLOG_DEBUG("Running do_dopamine_plasticity on {} blocks, {} threads", num_blocks, num_threads);
    cudaDeviceSynchronize();
    do_dopamine_plasticity_kernel<<<num_blocks, num_threads>>>(synapses, population.neurons_.mut_view(), step);
    cudaDeviceSynchronize();
    SPDLOG_DEBUG("Running update and renormalize.");
    update_and_renormalize<<<num_blocks, num_threads>>>(synapses, population.neurons_.mut_view(), step);
    cudaDeviceSynchronize();
    SPDLOG_DEBUG("Finished dopamine plasticity calculation.");
}
} // namespace knp::backends::gpu::cuda

using ResourceSynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceSynapseValue = typename knp::backends::gpu::cuda::CUDAProjection<ResourceSynapseType>::Synapse;
using ResourceSynapseValuePtr = ResourceSynapseValue*;

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeIndex);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(ResourceSynapseValue);
REGISTER_CUDA_VECTOR_TYPE(ResourceSynapseValuePtr);
