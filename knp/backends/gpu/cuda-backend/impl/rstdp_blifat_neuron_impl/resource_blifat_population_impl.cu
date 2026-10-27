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

#include "../backend_impl.cuh"
#include "../cuda_lib/register_type.cuh"
#include "../population.cuh"
#include "../projection.cuh"
#include "../resource_delta_projection.cuh"
#include "../populations_impl.cuh"
#include "resource_stdp_impl.cuh"
#include <knp/core/population.h>

#include <vector>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
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


template<>
__global__ void calculate_neurons_impacts<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>(
        device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
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
            atomicAdd(&neuron.dopamine_value_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            neuron.total_blocking_period_ = static_cast<int64_t>(impact.impact_value_);
            break;
        default:
            // Non-implemented impact type, ignoring it.
            PRINTF_INFO("Unimlemented synapse type: %d.", static_cast<int>(impact.synapse_type_));
    }
    if (knp::synapse_traits::OutputType::EXCITATORY == impact.synapse_type_)
        neuron.is_being_forced_ |= is_forcing;
}


__global__ void calculate_neurons_post_impact(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                              SpikeIndex *spike_buffer, SpikeIndex *size_counter)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;
    bool spike = false;
    neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> &neuron = neurons.data_[neuron_index];
    // This logic probably needs to be changed in the future.
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


template <class SynapseType, int position_index>
device_lib::ValueIndex build_index_gpu_slow(const CUDAProjection<SynapseType> &gpu_projection)
{
    std::map<device_lib::LongIndex, std::vector<device_lib::LongIndex>> buffer;
    for (size_t i = 0; i < gpu_projection.synapses_.size(); ++i)
    {
        const auto &synapse = gpu_projection.synapses_.copy_at(i);
        const auto neuron_id = ::cuda::std::get<position_index>(synapse);
        auto map_iter = buffer.find(neuron_id);
        if (map_iter == buffer.end())
        {
            map_iter = buffer.insert(std::make_pair(neuron_id, std::vector<device_lib::LongIndex>{})).first;
        }
        map_iter->second.push_back(i);
    }
    device_lib::LongIndex current_offset = 0;
    const device_lib::LongIndex last_neuron = buffer.rbegin()->first;

    std::vector<device_lib::LongIndex> indices(gpu_projection.synapses_.size());
    std::vector<device_lib::LongIndex> offsets;
    offsets.reserve(last_neuron + 2);  // Neuron number + 1 is last_index + 2.
    offsets.push_back(0);
    device_lib::LongIndex first_neuron = 0;
    for (auto iter = buffer.begin(); iter != buffer.end(); ++iter)
    {
        const device_lib::LongIndex current_neuron = iter->first;
        // Filling offsets for skipped neurons: if first three are missing that would be 0, 0, 0, 0, 5...
        for (auto i = first_neuron; i < current_neuron; ++i)
        {
            offsets.push_back(current_offset);
        }
        first_neuron = current_neuron + 1;
        std::copy(iter->second.begin(), iter->second.end(), indices.begin() + current_offset);
        current_offset += iter->second.size();
        offsets.push_back(current_offset);
    }
    device_lib::ValueIndex index;
    index.indices_ = indices;
    index.offsets_ = offsets;
    return index;
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
        SPDLOG_INFO("No working projections found for a population \"{}\"", p_name.c_str());
    }

    using ResourceProjection = CUDAProjection<synapse_traits::SynapticResourceSTDPDeltaSynapse>;
    SpikeIndex size = 0;
    cudaMemcpy(&size, counter, sizeof(SpikeIndex), cudaMemcpyDeviceToHost);
    cudaFree(counter);
    device_lib::CUDAVector<SpikeIndex> result{output, size};

    if (working_projection_indices.size() != 0)
    {
        auto &projection_var = this_backend->get_projection(working_projection_indices[0]);

        constexpr int type_index = boost::mp11::mp_find<CUDABackend::SupportedSynapses, ResourceSynapseType>();
        ResourceProjection *projection_ptr = ::cuda::std::get_if<type_index>(&projection_var);
        if (!projection_ptr)
        {
            SPDLOG_ERROR("Wrong projection type when extracting");
            throw std::runtime_error("Wrong type of projection extraction");
        }
        if (projection_ptr->synapses_by_postsynaptic_.synapses_.size() == 0)
        {
            auto index = build_index_gpu_slow<RSTDPDeltaSynapse, core::target_neuron_id>(*projection_ptr);
            projection_ptr->synapses_by_postsynaptic_ = initialize_synapses_per_neurons(index.view(),
                    projection_ptr->synapses_.data());
        }
        auto synapses = projection_ptr->synapses_by_postsynaptic_.view();
        update_rstdp_synapses(population, synapses, result, step);
    }
    return result;
}

} // namespace knp::backends::gpu::cuda

using ResourceSynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceSynapseValue = typename knp::backends::gpu::cuda::CUDAProjection<ResourceSynapseType>::Synapse;
using ResourceSynapseValuePtr = ResourceSynapseValue*;

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeIndex);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(ResourceSynapseValue);
REGISTER_CUDA_VECTOR_TYPE(ResourceSynapseValuePtr);
