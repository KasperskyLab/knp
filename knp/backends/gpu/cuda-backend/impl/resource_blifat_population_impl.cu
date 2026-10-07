/**
 * @file resource_blifat_population.cuh
 * @brief Contains functions for population calculation.
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

#include "backend_impl.cuh"
#include "population.cuh"
#include "projection.cuh"
#include "resource_delta_projection.cuh"
#include "populations_impl.cuh"
#include <knp/core/population.h>

#include <vector>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
using ResourceBlifatParams = knp::neuron_traits::neuron_parameters<
        knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceSynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using SynapseValue = typename CUDAProjection<ResourceSynapseType>::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapseType>;


__global__ void calculate_neurons_pre_impact(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                             StepIndex current_step)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;
    ResourceBlifatParams &neuron = neurons.data_[neuron_index];
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    neuron.dopamine_value_ = 0.0;
    neuron.is_being_forced_ = false;

    if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
    {
        neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
    }
    else
    {
        neuron.potential_ *= neuron.potential_decay_;
    }
    neuron.pre_impact_potential_ = neuron.potential_;
}


__global__ void calculate_neurons_impacts(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                          device_lib::CUDAVectorView <SynapticImpact> impacts, bool is_forcing)
{
    const size_t impact_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (impact_index >= impacts.size_) return;
    const SynapticImpact &impact = impacts.data_[impact_index];
    if (impact.postsynaptic_neuron_index_ >= neurons.size_) return;
    auto &neuron = neurons.data_[impact.postsynaptic_neuron_index_];
    switch (impact.synapse_type_)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            atomicAdd(&neuron.potential_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            atomicAdd(&neuron.potential_, -impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
            atomicAdd(&neuron.inhibitory_conductance_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::DOPAMINE:
            printf("Adding dopamine %f to neuron %u\n", impact.impact_value_, impact.postsynaptic_neuron_index_);
            atomicAdd(&neuron.dopamine_value_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            neuron.total_blocking_period_ = static_cast<int64_t>(impact.impact_value_);
            break;
    }
    if (knp::synapse_traits::OutputType::EXCITATORY == impact.synapse_type_)
        neuron.is_being_forced_ |= is_forcing;
}


// TODO: template this, it's a complete copy from populations_impl.cu
__host__ void calculate_neurons_impacts_all(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                            const std::vector <SynapticImpactMessage> &messages_all,
                                            std::vector <device_lib::LongIndex> message_ids)
{
    for (const auto &msg_id: message_ids)
    {
        const auto &msg = messages_all[msg_id];
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.impacts_.size());
        calculate_neurons_impacts<<<num_blocks, num_threads>>>(neurons, msg.impacts_.view(), msg.is_forcing_);
    }
    cudaDeviceSynchronize();
}


__global__ void calculate_neurons_post_impact(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                              SpikeIndex *spike_buffer, SpikeIndex *size_counter)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;
    bool spike = false;
    neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> &neuron = neurons.data_[neuron_index];
    if (neuron.total_blocking_period_ <= 0)
    {
        // Restore potential that the neuron had before impacts.
        neuron.potential_ = neuron.pre_impact_potential_;
        bool was_negative = neuron.total_blocking_period_ < 0;
        // If it is negative, increase by 1.
        neuron.total_blocking_period_ += was_negative;
        // If it is now zero, but was negative before, increase it to max, else leave it as is.
        neuron.total_blocking_period_ +=
                std::numeric_limits<int64_t>::max() * ((neuron.total_blocking_period_ == 0) && was_negative);
    }
    else
    {
        neuron.total_blocking_period_ -= 1;
    }

    if (neuron.inhibitory_conductance_ < 1.0)
    {
        neuron.potential_ -=
                (neuron.potential_ - neuron.reversal_inhibitory_potential_) *
                neuron.inhibitory_conductance_;
    }
    else
    {
        neuron.potential_ = neuron.reversal_inhibitory_potential_;
    }

    if ((neuron.n_time_steps_since_last_firing_ > neuron.absolute_refractory_period_) &&
        (neuron.potential_ >= neuron.activation_threshold_ + neuron.dynamic_threshold_))
    {
        // Spike.
        neuron.dynamic_threshold_ += neuron.threshold_increment_;
        neuron.postsynaptic_trace_ += neuron.postsynaptic_trace_increment_;

        neuron.potential_ = neuron.potential_reset_value_;
        neuron.bursting_phase_ = neuron.bursting_period_;
        neuron.n_time_steps_since_last_firing_ = 0;
        spike = true;
    }

    if (neuron.potential_ < neuron.min_potential_)
    {
        neuron.potential_ = neuron.min_potential_;
    }
    if (spike)
    {
        SpikeIndex counter = atomicAdd(size_counter, 1);
        spike_buffer[counter] = neuron_index;
    }
}


struct SynapsesPerNeurons
{
    device_lib::LongIndex offsets_size_;
    device_lib::LongIndex *offsets_;

    SynapseValue **synapses_;
    device_lib::LongIndex synapses_size_;
};


__global__ void index_to_pointer(device_lib::IndexView synapse_index, SynapseValue *start, SynapseValue **output)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapse_index.indices_size_) return;
    output[synapse_id] = start + synapse_index.indices_ptr_[synapse_id];
}


__device__ device_lib::CUDAVectorMutableView<SynapseValue*>
    extract_synapses_from_index(
        SynapsesPerNeurons &synapse_index, device_lib::LongIndex neuron_index)
{
    assert(synapse_index.offsets_size_ > 0);
    device_lib::LongIndex offset = synapse_index.offsets_[neuron_index];
    device_lib::LongIndex size = synapse_index.offsets_[neuron_index + 1] - offset;
    return device_lib::CUDAVectorMutableView<SynapseValue*>{synapse_index.synapses_ + offset, size};
}


__device__ void recalculate_synapse_weight(ResourceSynapseParams &synapse)
{
    const auto syn_w = std::max(synapse.rule_.synaptic_resource_, 0.F);
    const auto weight_diff = synapse.rule_.w_max_ - synapse.rule_.w_min_;
    const auto weight_before = synapse.weight_;
    synapse.weight_ = synapse.rule_.w_min_ + weight_diff * syn_w / (weight_diff + syn_w);
    if (abs(weight_before - synapse.weight_) >= 1e-5)
        printf("Weight before %f, after %f\n", weight_before, synapse.weight_);
}


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
            if (neuron.last_step_ - step < neuron.isi_max_)
            {
                neuron.isi_status_ = neuron_traits::ISIPeriodType::period_continued;
            }
            break;
        case neuron_traits::ISIPeriodType::period_continued:
            if (neuron.last_step_ - step >= neuron.isi_max_ || neuron.dopamine_value_ != 0)
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
                                                   StepIndex step)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_) return;
    ResourceSynapseParams &synapse = ::cuda::std::get<0>(*synapses.data_[synapse_id]);
    float threshold_update = synapse.weight_ * (synapse.weight_ > 0) * neuron->synapse_sum_threshold_coefficient_;
    // technically a race condition, float operations are not order-independent, but oh well.
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
        // TODO: NOT HERE. This shouldn't matter now as d_u_ is zero for our task, but the logic is wrong.
        printf("1. Substracting from synapse resource %f value %f\n", synapse.rule_.synaptic_resource_,
               synapse.rule_.d_u_);
        synapse.rule_.synaptic_resource_ -= synapse.rule_.d_u_;
        atomicAdd(&neuron->free_synaptic_resource_, synapse.rule_.d_u_);
        // Hebbian plasticity.
        // 1. Check if synapse ever got a spike in the current ISI period.
        if (synapse.rule_.has_contributed_ && !synapse.rule_.had_hebbian_update_)
        {
            // printf("Synapse %lu contributed\n", synapse_id);
            // 2. If it did, then update synaptic resource value.
            const float d_h = neuron->d_h_
                    * ::cuda::std::min(static_cast<float>(::cuda::std::pow(2, -neuron->stability_)), 1.F);
            printf("Adding to synaptic resource %f value %f, neuron dh %f\n", synapse.rule_.synaptic_resource_, d_h,
                   neuron->d_h_);
            atomicAdd(&synapse.rule_.synaptic_resource_, d_h);
            atomicAdd(&neuron->free_synaptic_resource_, -d_h);
            synapse.rule_.had_hebbian_update_ = true;
        }
        // Recalculating synapse weights. Sometimes it probably doesn't need to happen, check it later.
    }
    recalculate_synapse_weight(synapse);
}


__host__ SynapsesPerNeurons initialize_synapses_per_neurons(
        const device_lib::IndexView &synapse_index,
        SynapseValue *start)
{
    SynapsesPerNeurons result;
    SPDLOG_DEBUG("Initializing synapses_per_neurons, index:");
    call_and_check(cudaMalloc(&result.offsets_, sizeof(device_lib::LongIndex) * synapse_index.offsets_size_));
    call_and_check(cudaMalloc(&result.synapses_, sizeof(void*) * synapse_index.indices_size_));
    // TODO static_assert(is_same_type(SynapsesPerNeurons::synapses_, ValueIndexView::offsets_))
    cudaMemcpy(result.offsets_, synapse_index.offsets_ptr_, sizeof(device_lib::LongIndex) * synapse_index.offsets_size_,
               cudaMemcpyDeviceToDevice);
    result.offsets_size_ = synapse_index.offsets_size_;
    result.synapses_size_ = synapse_index.indices_size_;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapse_index.indices_size_);
    index_to_pointer<<<num_blocks, num_threads>>>(synapse_index, start, result.synapses_);
    cudaDeviceSynchronize();
    return result;
}


__global__ void process_spiking_neurons_kernel(device_lib::CUDAVectorView<SpikeIndex> spikes,
                                               device_lib::CUDAVectorMutableView<ResourceBlifatParams> neurons,
                                               SynapsesPerNeurons synapses,
                                               StepIndex step)
{
    const device_lib::LongIndex spike_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (spike_id >= spikes.size_) return;
    const auto neuron_id = spikes.data_[spike_id];
    ResourceBlifatParams &neuron = neurons.data_[neuron_id];
    printf("Neuron id %u, neuron dh %f\n", neuron_id, neuron.d_h_);
    neuron.last_spike_step_ = step;
    // Calculate neuron ISI status.
    update_isi(neuron, step);
    if (knp::neuron_traits::ISIPeriodType::period_started == neuron.isi_status_)
        neuron.stability_ -= neuron.stability_change_at_isi_;
    neuron.additional_threshold_ = 0.0;
    device_lib::CUDAVectorMutableView<SynapseValue*> synapse_view = extract_synapses_from_index(synapses, neuron_id);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(synapse_view.size_);
    process_synapses_per_neuron_kernel<<<num_blocks, num_threads>>>(&neuron, synapse_view, step);
}


void process_spiking_neurons_impl(const device_lib::CUDAVector<SpikeIndex> &msg, SynapsesPerNeurons &synapse_index,
                                  device_lib::CUDAVector<ResourceBlifatParams> &neurons, StepIndex step)
{
    // It's very important that during this function no projection invalidates iterators.
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.size());
    process_spiking_neurons_kernel<<<num_blocks, num_threads>>>(msg.view(), neurons.mut_view(), synapse_index, step);
}


// TODO: Merge synapses pointers
//__host__ SynapsesPerNeurons merge_synapses_per_neurons(const SynapsesPerNeurons **synapses_array)
//{
//    // result.offsets[i] = sum(array.offsets[i])
//    // result.synapses[offsets[i]] = concat(array.synapses[array[i].offsets[i] : array[i].offsets[i + 1]])
//}


__global__ void add_resource_to_synapses(device_lib::CUDAVectorMutableView<SynapseValue*> synapses,
                                         double add_resource_value)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_) return;
    auto &synapse = ::cuda::std::get<0>(*synapses.data_[synapse_id]);
    printf("3. Adding to synapse resource %f value %f\n", synapse.rule_.synaptic_resource_, add_resource_value);
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
    // printf("Starting do_dopamine_plasticity_kernel for nonzero dopamine value\n");
    auto &synapse = ::cuda::std::get<0>(*synapses.data_[synapse_id]);
//    if (synapse.rule_.has_contributed_) printf("Step %lu, neuron last step %lu, plasticity time %u\n",
//                                               step, neuron->last_spike_step_, neuron->dopamine_plasticity_time_);
    if (step - neuron->last_spike_step_ <= neuron->dopamine_plasticity_time_ && synapse.rule_.has_contributed_)
    {
        // Change synapse resource.
        float resource_change =
                neuron->dopamine_value_ * ::cuda::std::min(static_cast<float>(std::pow(2, -neuron->stability_)), 1.F);
        printf("Changing synapse %lu resource by %f\n", synapse_id, resource_change);
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


__global__ void do_dopamine_plasticity_kernel(SynapsesPerNeurons synapse_pointer_index,
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
        printf("Step %lu, Neuron %lu, dopamine value %f, num synapses %lu\n", step, neuron_id,
               neurons.data_[neuron_id].dopamine_value_,
               extract_synapses_from_index(synapse_pointer_index, neuron_id).size_);

        do_dopamine_plasticity_device<ResourceSynapseType>(
                extract_synapses_from_index(synapse_pointer_index, neuron_id),
                neurons.data_[neuron_id], step);
    }
}


__global__ void update_and_renormalize(SynapsesPerNeurons synapse_pointer_index,
                                       device_lib::CUDAVectorMutableView<ResourceBlifatParams> neurons, StepIndex step)
{
    const device_lib::LongIndex neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= neurons.size_) return;
    update_neuron_stability(neurons.data_[neuron_id], step);
    if (synapse_pointer_index.offsets_size_ == 0 || neuron_id >= synapse_pointer_index.offsets_size_ - 1) return;
    renormalize_resource(extract_synapses_from_index(synapse_pointer_index, neuron_id), neurons.data_[neuron_id], step);
}


device_lib::CUDAVector<SpikeIndex> calculate_population(
        CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population, CUDABackendImpl *this_backend,
        StepIndex step)
{
    auto [num_blocks_neuro, num_threads_neuro] = device_lib::get_blocks_config(population.neurons_.size());

    calculate_neurons_pre_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), step);
    auto &device_message_bus = this_backend->get_message_bus();
    std::vector<device_lib::LongIndex> message_ids
            = device_message_bus.unload_messages<cuda::SynapticImpactMessage>(population.uid_);

    if (!message_ids.empty())
    {
        SPDLOG_DEBUG("Running calculate impacts on {} messages", message_ids.size());
        calculate_neurons_impacts_all(population.neurons_.mut_view(),
                                      device_message_bus.all_messages<SynapticImpactMessage>(),
                                      message_ids);
    }
    // TODO TEMP
    SpikeIndex *output;
    cudaMalloc(&output, sizeof(SpikeIndex) * population.neurons_.size());
    SpikeIndex *counter;
    cudaMalloc(&counter, sizeof(SpikeIndex));
    cudaMemset(counter, 0, sizeof(SpikeIndex));
    calculate_neurons_post_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), output,
                                                                           counter);

    std::vector<device_lib::LongIndex> working_projection_indices = this_backend->find_projections_by_postsynaptic<
            synapse_traits::SynapticResourceSTDPDeltaSynapse>(population.uid_, true);
    if (working_projection_indices.size() == 0)
    {
        const auto p_name = population.tags_.exists("name") ?
                                     population.tags_.get_tag<std::string>("name") : "unnamed";
        SPDLOG_WARN("No working projections found for a population \"{}\"", p_name.c_str());
    }

    using ResourceProjection = CUDAProjection<synapse_traits::SynapticResourceSTDPDeltaSynapse>;
    // We have a number of projections, let's take them and for each we have a VectorView with synapses per neuron.
    // So it's a VectorView<VectorView<LongIndex>>? No. We need a CUDAVector of synapse pointers. Per neuron? No.
    // We start a kernel for each working projection + population, that gives us a set(?) of vectors views per
    // projection. Number of vecviews is equal to the number of neurons, size of set is the number of working
    // projections. So. We create a vector of views... This is a really complex structure, actually!
    // Let's try to simplify at least somehow. What we need is a VectorView of synapse pointers per neuron. Or not,
    // maybe more like a multivector? Nah, a flat container should be better. So, offsets and synapses?
    // Let's start with the result. The result is: a long list of synapse pointer and a shorter list of offsets per
    // neuron. We can make one of those per projection, easily.

    SpikeIndex size = 0;
    cudaMemcpy(&size, counter, sizeof(SpikeIndex), cudaMemcpyDeviceToHost);
    cudaFree(counter);
    device_lib::CUDAVector<SpikeIndex> result{output, size};

    if (working_projection_indices.size() != 0)
    {
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(population.neurons_.size());
        auto &projection_var = this_backend->get_projection(working_projection_indices[0]);

        constexpr int type_index = boost::mp11::mp_find<SupportedSynapses, ResourceSynapseType>();
        ResourceProjection *projection_ptr = ::cuda::std::get_if<type_index>(&projection_var);
        if (!projection_ptr)
        {
            SPDLOG_ERROR("Wrong projection type when extracting");
            throw std::runtime_error("Wrong type of projection extraction");
        }
        SynapsesPerNeurons synapses = initialize_synapses_per_neurons(projection_ptr->index_by_postsynaptic_.view(),
                                                                      projection_ptr->synapses_.data());
        if (result.size())
        {
            process_spiking_neurons_impl(result, synapses, population.neurons_, step);
        }
        SPDLOG_WARN("Running do_dopamine_plasticity on {} blocks, {} threads", num_blocks, num_threads);
        do_dopamine_plasticity_kernel<<<num_blocks, num_threads>>>(synapses, population.neurons_.mut_view(), step);
        update_and_renormalize<<<num_blocks, num_threads>>>(synapses, population.neurons_.mut_view(), step);

        cudaFree(synapses.offsets_);
        cudaFree(synapses.synapses_);
    }
    return result;
}

} // namespace knp::backends::gpu::cuda
