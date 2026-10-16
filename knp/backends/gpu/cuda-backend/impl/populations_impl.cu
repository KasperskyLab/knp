/**
 * @file populations_impl.cu
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

#include "population.cuh"
#include "populations_impl.cuh"
#include <knp/core/population.h>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
using BlifatParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;


__global__ void calculate_neurons_pre_impact(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                             StepIndex current_step)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;

    BlifatParams &neuron = neurons.data_[neuron_index];
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    /*
    if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
    {
        neuron.dopamine_value_ = 0.0;
        neuron.is_being_forced_ = false;
    }
    */

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


__global__ void calculate_neurons_impacts(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                          device_lib::CUDAVectorView<SynapticImpact> impacts)
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
            neuron.total_blocking_period_ = static_cast<unsigned int>(impact.impact_value_);
            break;
    }
}


__host__ void calculate_neurons_impacts_all(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                            const std::vector<SynapticImpactMessage> &messages_all,
                                            std::vector<device_lib::LongIndex> message_ids)
{
    for (const auto &msg_id : message_ids)
    {
        const auto &msg = messages_all[msg_id];
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.impacts_.size());
        calculate_neurons_impacts<<<num_blocks, num_threads>>>(neurons, msg.impacts_.view());
    }
    cudaDeviceSynchronize();
}


__global__ void calculate_neurons_post_impact(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                              SpikeIndex *spike_buffer, SpikeIndex *size_counter)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    bool spike = false;
    neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> &neuron = neurons.data_[neuron_index];
    if (neuron.total_blocking_period_ <= 0)
    {
        // TODO: Make it more readable, don't be afraid to use if operators.
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


device_lib::CUDAVector<SpikeIndex> calculate_population(
        CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population, const CUDAMessageBus& device_message_bus,
        StepIndex step)
{
    auto [num_blocks_neuro, num_threads_neuro] = device_lib::get_blocks_config(population.neurons_.size());

    calculate_neurons_pre_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), step);
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

    SpikeIndex out_size = 0;
    cudaMemcpy(&out_size, counter, sizeof(SpikeIndex), cudaMemcpyDeviceToHost);
    // Capacity would be "out_size" while the ptr is larger, but that doesn't matter as the pointer is freed as a whole.
    device_lib::CUDAVector<SpikeIndex> result{output, out_size};
    cudaFree(counter);
    return result;
}
} // namespace knp::backends::gpu::cuda
