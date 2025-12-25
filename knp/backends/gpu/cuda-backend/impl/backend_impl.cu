/**
 * @file backend_impl.cu
 * @brief CUDABackendImpl backend class implementation.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
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


#include <knp/devices/gpu_cuda.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <spdlog/spdlog.h>

#include <limits>
#include <vector>

#include <boost/mp11.hpp>

#include <algorithm>

#include "backend_impl.cuh"
#include "projection.cuh"
#include "population.cuh"

#include "cuda_lib/get_blocks_config.cuh"
#include "cuda_lib/printf.cuh"
#include "cuda_lib/vector.cuh"
#include "cuda_bus/messaging.cuh"


namespace knp::backends::gpu::cuda
{
// helper type for the visitor.
template<class... Ts>
struct overloaded : Ts ...
{
    using Ts::operator()...;
};
// explicit deduction guide.
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;


template<class ProjectionType>
__host__ __device__ inline bool is_forcing()
{
    return false;
}


template <>
CUDABackendImpl::PopulationVariants gpu_extract<CUDABackendImpl::PopulationVariants>(
    const CUDABackendImpl::PopulationVariants *);

template <>
void gpu_insert<CUDABackendImpl::PopulationVariants>(const CUDABackendImpl::PopulationVariants &,
                                                     CUDABackendImpl::PopulationVariants *);

template <>
CUDABackendImpl::ProjectionVariants gpu_extract<CUDABackendImpl::ProjectionVariants>(
        const CUDABackendImpl::ProjectionVariants *);

template <>
void gpu_insert<CUDABackendImpl::ProjectionVariants>(const CUDABackendImpl::ProjectionVariants &,
                                                     CUDABackendImpl::ProjectionVariants *);

namespace detail
{
    template <class Variant, class Instance>
    __global__ void make_variant_kernel(Variant *result, Instance *source)
    {
        new (result) Variant(*source);
    }
}


template<class TypeVariant, size_t index>
TypeVariant extract_by_index(const void *type_ptr)
{
    return gpu_extract<boost::mp11::mp_at_c<TypeVariant, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<TypeVariant, index> *>(type_ptr));
}


// TODO: Make a template, it's also used for messages.

template<typename T>
__host__ __device__ void get_kernel(const T *var, int *type, const void **val)
{
    int type_val = var->index();
    static_assert(::cuda::std::variant_size<T>() == 1, "Incorrect variant size!");
    switch (type_val)
    {
        case 0:
            *val = ::cuda::std::get_if<0>(var);
            break;
        default:
            *val = nullptr;
    }
    *type = type_val;
}


__global__ void get_population_kernel(const CUDABackendImpl::PopulationVariants *var, int *type, const void **pop)
{
    get_kernel(var, type, pop);
}


__global__ void get_projection_kernel(const CUDABackendImpl::ProjectionVariants *var, int *type, const void **proj)
{
    get_kernel(var, type, proj);
}


template<>
void gpu_insert<CUDABackendImpl::PopulationVariants>(const CUDABackendImpl::PopulationVariants &cpu_source,
                                                     CUDABackendImpl::PopulationVariants *gpu_target)
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


template<>
void gpu_insert<CUDABackendImpl::ProjectionVariants>(const CUDABackendImpl::ProjectionVariants &cpu_source,
                                                     CUDABackendImpl::ProjectionVariants *gpu_target)
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


template<>
__host__ __device__ inline bool is_forcing<cuda::CUDAProjection<synapse_traits::DeltaSynapse>>()
{
    return true;
}


template<class T>
__global__ void get_uids_kernel(const T *data, size_t size, cuda::UID *result)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    result[index] = ::cuda::std::visit([](auto &v) { return v.uid_; }, data[index]);
}


template<class VectorData>
device_lib::CUDAVector<cuda::UID> get_uids(const device_lib::CUDAVector<VectorData> &entities)
{
    device_lib::CUDAVector<cuda::UID> result(entities.size());
    if (entities.size() != 0)
    {
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(entities.size());
        get_uids_kernel<<<num_blocks, num_threads>>>(entities.data(), entities.size(), result.data());
    }
    return result;
}


/// @note: out_messages_data should be of size to contain at least num_populations messages.
__global__ void calculate_populations_kernel(CUDABackendImpl::PopulationVariants *populations, size_t num_populations,
                                             const cuda::MessageVariant *messages, size_t messages_size,
                                             const cuda::device_lib::CUDAVector<uint64_t> *indices, size_t indices_size,
                                             cuda::MessageVariant *out_messages_data, std::uint64_t step)
{
    // Calculate populations. This is the same as inference.
    size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= num_populations) return;

    CUDABackendImpl::PopulationVariants &population = populations[thread_index];
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> new_messages;
    PRINTF_TRACE("Population index: %lu\n", population.index());

    size_t num_messages = indices[thread_index].size();
    for (size_t n = 0; n < num_messages; ++n)
    {
        uint64_t message_index = indices[thread_index][n];
        if (message_index >= messages_size) continue;
        PRINTF_TRACE("Messages size: %lu, message index: %lu\n", messages_size, message_index);
        new_messages.push_back(messages[message_index]);
    }

    auto message = ::cuda::std::visit([&new_messages, step](auto &pop)
    {
        return CUDABackendImpl::calculate_population(pop, new_messages, step);
    }, population);

    if (message) out_messages_data[thread_index] = cuda::MessageVariant{message.value()};
}


void CUDABackendImpl::calculate_populations(std::uint64_t step)
{
    // Calculate populations. This is the same as inference.
    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
    if (!device_populations_.size()) return;

    device_lib::CUDAVector<cuda::UID> population_uids = get_uids(device_populations_);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(device_populations_.size());

    device_lib::CUDAVector<device_lib::CUDAVector<uint64_t>> population_messages(device_populations_.size());

    for (size_t i = 0; i < device_populations_.size(); ++i)
    {
        const device_lib::CUDAVector<uint64_t> message_ids = device_message_bus_.unload_messages<SynapticImpactMessage>(
                population_uids.copy_at(i));
        gpu_insert(message_ids, population_messages.data() + i);
    }

    MessageVector out_messages(device_populations_.size());
    assert(device_populations_.size() == population_messages.size());
    calculate_populations_kernel<<<num_blocks, num_threads>>>(device_populations_.data(), device_populations_.size(),
                                                              device_message_bus_.all_messages().data(),
                                                              device_message_bus_.all_messages().size(),
                                                              population_messages.data(), population_messages.size(),
                                                              out_messages.data(), step);
    cudaDeviceSynchronize();
    device_message_bus_.send_message_gpu_batch(out_messages);
}


/**
 * Calculate a step for all projections in a network.
 * @param projections vector of projection variants.
 * @param num_projections number of projections.
 * @param messages a vector of all messages in the GPU bus.
 * @param messages_size number of all messages in the GPU bus.
 * @param indices indices of messages directed at each projection.
 * @param step current network step.
 * @note make sure number of valid indices is equal to num_projections.
 */
__global__ void
calculate_projections_kernel(CUDABackendImpl::ProjectionVariants *projections, size_t num_projections,
                             const cuda::MessageVariant *messages, size_t messages_size,
                             const cuda::device_lib::CUDAVector<uint64_t> *indices,
                             std::uint64_t step)
{
    // Calculate projections.
    // using namespace ::cuda::std::placeholders;
    PRINTF_TRACE("prjs: %lu %p\n msgs: %lu %p\n inds: %p\n", num_projections, projections, messages_size, messages,
           indices);
    size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= num_projections) return;

    CUDABackendImpl::ProjectionVariants &projection = projections[thread_index];
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> msgs; // (indices[thread_index].size());
    for (size_t n = 0; n < indices[thread_index].size(); ++n)  // Almost always 1 or 0 iterations.
    {
        uint64_t message_index = indices[thread_index][n];
        if (message_index >= messages_size) continue;
        PRINTF_TRACE("size %lu message_index %lu, n %lu\n", msgs.size(), message_index, n);
        msgs.push_back(messages[message_index]);
        PRINTF_TRACE("Msgs after adding: %lu\n", msgs.size());
    }
    ::cuda::std::visit([&msgs, step](auto &proj)
    {
        CUDABackendImpl::calculate_projection(proj, msgs, step);
    }, projection);
}


void CUDABackendImpl::calculate_projections(std::uint64_t step)
{
    // Calculate projections.
    device_lib::CUDAVector<cuda::UID> projection_uids = get_uids(device_projections_);

    if (!device_projections_.size()) return;

    device_lib::CUDAVector<device_lib::CUDAVector<uint64_t>> projection_messages(device_projections_.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const device_lib::CUDAVector<uint64_t> message_ids = device_message_bus_.unload_messages<SpikeMessage>(
                projection_uids.copy_at(i));
        gpu_insert(message_ids, projection_messages.data() + i);
    }

    auto [num_blocks, num_threads] = device_lib::get_blocks_config(device_projections_.size());
    assert(device_projections_.size() == projection_messages.size());
    calculate_projections_kernel<<<num_blocks, num_threads>>>(device_projections_.data(),
                                                              device_projections_.size(),
                                                              device_message_bus_.all_messages().data(),
                                                              device_message_bus_.all_messages().size(),
                                                              projection_messages.data(),
                                                              step);
    cudaDeviceSynchronize();
}


void CUDABackendImpl::load_populations(const knp::backends::gpu::CUDABackend::PopulationContainer &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());

    device_populations_.clear();
    device_populations_.reserve(populations.size());

    for (const auto &population : populations)
    {
        ::std::visit([this](auto &arg)
        {
            using CPUPopulationType = std::decay_t<decltype(arg)>;

            auto pop = CUDAPopulation<typename CPUPopulationType::PopulationNeuronType>(arg);
            device_populations_.push_back(pop);
        }, population);
    }

    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackendImpl::load_projections(const knp::backends::gpu::CUDABackend::ProjectionContainer &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());

    device_projections_.clear();
    device_projections_.reserve(projections.size());

    for (const auto &projection : projections)
    {
        ::std::visit([this](auto &arg)
        {
            using CPUProjectionType = std::decay_t<decltype(arg)>;

            auto proj = CUDAProjection<typename CPUProjectionType::ProjectionSynapseType>{arg};
            device_projections_.push_back(proj);
        }, projection);
    }

    SPDLOG_DEBUG("All projections loaded.");
}


__global__ void get_projection_uids_kernel(const CUDABackendImpl::ProjectionVariants *projection,
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


auto get_projection_uids(const CUDABackendImpl::ProjectionVariants *proj)
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


void CUDABackendImpl::init()
{
    SPDLOG_DEBUG("Initializing CUDABackendImpl...");

    // knp::backends::cpu::init(projections_, get_message_endpoint());
    // for (const auto &p : device_projections_)
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const auto [pre_uid, post_uid, this_uid] = get_projection_uids(device_projections_.data() + i);
        if (!cuda::empty_uid(pre_uid)) this->device_message_bus_.subscribe_gpu<cuda::SpikeMessage>(this_uid, {pre_uid});
        if (!cuda::empty_uid(post_uid))
        {
            this->device_message_bus_.subscribe_gpu<cuda::SynapticImpactMessage>(post_uid, {this_uid});
        }
    }

    SPDLOG_DEBUG("Initialization finished.");
}


__device__ ::cuda::std::optional<knp::backends::gpu::cuda::SpikeMessage> CUDABackendImpl::calculate_population(
    CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population,
    const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
    std::uint64_t step_n)
{
    constexpr size_t spike_message_index =
        boost::mp11::mp_find<cuda::MessageVariant, cuda::SynapticImpactMessage>();

    // TODO rework
    for (size_t i = 0; i < population.neurons_.size(); ++i)
    {
        neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> neuron = population.neurons_[i];
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

        population.neurons_[i] = neuron;
    }

    // process_inputs(population, messages);
    for (const knp::backends::gpu::cuda::MessageVariant &message_var : messages)
    {
        if (message_var.index() != spike_message_index) continue;
        const SynapticImpactMessage &message = ::cuda::std::get<SynapticImpactMessage>(message_var);

        for (size_t i = 0; i < message.impacts_.size(); ++i)
        {
            const auto &impact = message.impacts_[i];

            neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> neuron =
                    population.neurons_[impact.postsynaptic_neuron_index_];

            // impact_neuron<BlifatLikeNeuron>(neuron, impact.synapse_type_, impact.impact_value_);
            switch (impact.synapse_type_)
            {
                case knp::synapse_traits::OutputType::EXCITATORY:
                    neuron.potential_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
                    neuron.potential_ -= impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
                    neuron.inhibitory_conductance_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::DOPAMINE:
                    neuron.dopamine_value_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::BLOCKING:
                    neuron.total_blocking_period_ = static_cast<unsigned int>(impact.impact_value_);
                    break;
            }

            /*if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
            {
                if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
                {
                    neuron.is_being_forced_ |= message.is_forcing_;
                }
            }*/
            population.neurons_[impact.postsynaptic_neuron_index_] = neuron;
        }
    }

    device_lib::CUDAVector<uint32_t> neuron_indexes;

    // calculate_neurons_post_input_state(population, neuron_indexes);
    for (size_t index = 0; index < population.neurons_.size(); ++index)
    {
        bool spike = false;
        neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> neuron = population.neurons_[index];

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

        // process_inputs(population, messages);
        for (const knp::backends::gpu::cuda::MessageVariant &message_var : messages)
        {
            if (message_var.index() != spike_message_index) continue;
            const SynapticImpactMessage &message = ::cuda::std::get<SynapticImpactMessage>(message_var);

            for (size_t i = 0; i < message.impacts_.size(); ++i)
            {
                const auto &impact = message.impacts_[i];
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
                    neuron_indexes.push_back(index);
                }

                population.neurons_[index] = neuron;
            }

            if (!neuron_indexes.empty())
            {
                cuda::SpikeMessage res_message
                        {
                                .header_ = {.sender_uid_ = population.uid_, step_n},
                                .neuron_indexes_ = neuron_indexes
                        };

                // device_message_bus_.send_message(res_message);
                return res_message;
            }

            return {};
        }
    }

    return {};
}


__device__ int64_t find_projection_messages(const CUDABackendImpl::ProjectionVariants *projection, uint64_t step)
{
    auto res = ::cuda::std::visit([step](const auto &proj) -> uint64_t
    {
        // TODO Parallelize
        for (uint64_t i = 0; i < proj.messages_.size(); ++i)
        {
            if (proj.messages_[i].header_.send_time_ == step) return i;  //! future_step?
        }

        return static_cast<uint64_t>(0u);
    }, *projection);

    return res;
}


__global__ void extract_projection_message(CUDABackendImpl::ProjectionVariants *projection, uint64_t step,
                                           device_lib::CUDAVector<MessageVariant> *messages_out)
{
    new (messages_out) device_lib::CUDAVector<MessageVariant>(0);
    ::cuda::std::visit([messages_out, step](auto &proj)
        {
            for (uint64_t i = 0; i < proj.messages_.size(); ++i)
            {
                if (proj.messages_[i].header_.send_time_ == step)
                {
                    // Sending message
                    messages_out->push_back(::cuda::std::move(proj.messages_[i]));
                    auto iter = proj.messages_.data() + i;
                    proj.messages_.erase(iter, iter + 1);
                }
            }
        }, *projection);
}




//__host__ uint64_t CUDABackendImpl::route_population_messages(uint64_t step)
//{
//    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
//    MessageVector *messages;
//    for (size_t i = 0; i < device_projections_.size(); ++i)
//    {
//        cudaMalloc(&messages, sizeof(MessageVector));
//        extract_population_message<<<1, 1>>>(device_populations_.data() + i, step, messages);
//        MessageVector msg_vec = gpu_extract<MessageVector>(messages);
//        message_bus_.send_message_gpu_batch(msg_vec);
//    }
//}


__host__ uint64_t CUDABackendImpl::route_projection_messages(uint64_t step)
{
    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
    MessageVector *messages;
    device_message_bus_.clear();

    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        cudaMalloc(&messages, sizeof(MessageVector));
        extract_projection_message<<<1, 1>>>(device_projections_.data() + i, step, messages);
        cudaDeviceSynchronize();
        MessageVector msg_vec = gpu_extract<MessageVector>(messages);
        device_message_bus_.send_message_gpu_batch(msg_vec);
    }

    return 0;
}


__device__ void CUDABackendImpl::calculate_projection(
    CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection,
    const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
    std::uint64_t step_n)
{
    constexpr size_t spike_message_index = boost::mp11::mp_find<cuda::MessageVariant, cuda::SpikeMessage>();
    PRINTF_TRACE("Messages size: %lu\n", messages.size());
    for (const knp::backends::gpu::cuda::MessageVariant &message_var : messages)
    {
        if (message_var.index() != spike_message_index) continue;
        const SpikeMessage &message = ::cuda::std::get<SpikeMessage>(message_var);
        PRINTF_TRACE("Processing message\n");
        const auto &message_data = message.neuron_indexes_;
        for (size_t i = 0; i < message_data.size(); ++i)
        {
            PRINTF_TRACE("Processing message data: index %lu, value %u\n", i, message_data[i]);
            const auto &spiked_neuron_index = message_data[i];
            PRINTF_TRACE("Projection size: %lu\n", projection.synapses_.size());
            for (size_t synapse_index = 0; synapse_index < projection.synapses_.size(); ++synapse_index)
            {
                CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse synapse =
                        projection.synapses_[synapse_index];
                if (::cuda::std::get<core::source_neuron_id>(synapse) != spiked_neuron_index) continue;
                const auto &synapse_params = ::cuda::std::get<core::synapse_data>(synapse);

                // The message is sent on step N - 1, received on step N. Step 0 delay 1 means the message is sent on 0.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                PRINTF_TRACE("Future step: %lu, delay: %u, weight: %f\n", future_step, synapse_params.delay_,
                       synapse_params.weight_);
                knp::backends::gpu::cuda::SynapticImpact impact{
                        synapse_index, synapse_params.weight_, synapse_params.output_type_,
                        static_cast<uint32_t>(::cuda::std::get<core::source_neuron_id>(synapse)),
                        static_cast<uint32_t>(::cuda::std::get<core::target_neuron_id>(synapse))};
                PRINTF_TRACE("Impact from neuron_%u to neuron_%u\n",
                       impact.presynaptic_neuron_index_,
                       impact.postsynaptic_neuron_index_);
                auto iter = projection.messages_.begin();
                // TODO: Easy to parallelize
                for (; iter != projection.messages_.end(); ++iter)
                {
                    if (iter->header_.send_time_ == future_step)
                    {
                        PRINTF_TRACE("Adding impact to existing message at future_step %lu\n", future_step);
                        iter->impacts_.push_back(impact);
                        break;
                    }
                }
                if (iter == projection.messages_.end())
                {
                    device_lib::CUDAVector<cuda::SynapticImpact> impacts(1);
                    impacts[0] = impact;
                    cuda::SynapticImpactMessage message_out{
                            {projection.uid_, future_step},
                            projection.presynaptic_uid_,
                            projection.postsynaptic_uid_,
                            ::cuda::std::move(impacts)};

                    message_out.is_forcing_ = is_forcing<cuda::CUDAProjection<synapse_traits::DeltaSynapse>>();
                    PRINTF_TRACE("Adding new_message to messages_ at step %lu\n", future_step);
                    projection.messages_.push_back(message_out);
                }
            }
        }
    }
}


__device__ void CUDABackendImpl::calculate_projection(
    CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
    const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
    std::uint64_t step_n)
{
    //SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
}


__device__ void CUDABackendImpl::calculate_projection(
    CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
    const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
    std::uint64_t step_n)
{
//    SPDLOG_TRACE("Calculate STDPSynapticResource synapse projection {}.", std::string(projection.get_uid()));
    // WeightUpdateSTDP<SynapseType>::init_synapse(std::get<core::synapse_data>(synapse), step_n);
    // Run:
    // knp::backends::cpu::calculate_delta_synapse_projection(
    //    projection, get_message_endpoint(), message_queue, get_step());


    // message_bus_.unload_messages<cuda::SpikeMessage>(projection.uid_, messages);

    // auto out_iter = calculate_delta_synapse_projection_data(projection, messages, future_messages, get_step());
    //
    // using SynapseType = typename ProjectionType::ProjectionSynapseType;
    // WeightUpdateSTDP<SynapseType>::init_projection(projection, messages, step_n);

                // WeightUpdateSTDP<SynapseType>::init_synapse(std::get<core::synapse_data>(synapse), step_n);
//                const auto &synapse_params = thrust::get<core::synapse_data>(synapse);
/*                const auto &synapse_params = thrust::get<core::synapse_data>(synapse);

                // The message is sent on step N - 1, received on step N.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                knp::backends::gpu::cuda::SynapticImpact impact{
                    synapse_index, synapse_params.weight_, synapse_params.output_type_,
                    static_cast<uint32_t>(thrust::get<core::source_neuron_id>(synapse)),
                    static_cast<uint32_t>(thrust::get<core::target_neuron_id>(synapse))};

                // ::cuda::std::find_if() is not implemented yet.
                auto iter = projection.messages_.begin();
                for (; iter != projection.messages_.end(); ++iter)
                {
                    if (iter->first == future_step) break;
                }

                if (iter != projection.messages_.end())
                {
                    iter->second.impacts_.push_back(impact);
                }
                else
                {
/*                    cuda::SynapticImpactMessage message_out{
                        {projection.uid_, step_n},
                        projection.presynaptic_uid_,
                        projection.postsynaptic_uid_,
                        is_forcing<cuda::CUDAProjection<synapse_traits::DeltaSynapse>>(),
                        {impact}};

                    projection.messages_.push_back(message_out));
*/
//                }
//            }
//        }
//    }

/*
    // WeightUpdateSTDP<SynapseType>::modify_weights(projection);
    return future_messages.find(step_n);
    //

    if (out_iter != future_messages.end())
    {
        // Send a message and remove it from the queue.
        message_bus_.send_message(out_iter->second);
        future_messages.erase(out_iter);
    }
*/
}


__host__ __device__ CUDABackendImpl::PopulationIterator CUDABackendImpl::begin_populations()
{
    return PopulationIterator{device_populations_.begin()};
}


__host__ __device__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::begin_populations() const
{
    return {device_populations_.cbegin()};
}


__host__ __device__ CUDABackendImpl::PopulationIterator CUDABackendImpl::end_populations()
{
    return PopulationIterator{device_populations_.end()};
}


__host__ __device__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::end_populations() const
{
    return device_populations_.cend();
}


__host__ __device__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::begin_projections()
{
    return ProjectionIterator{device_projections_.begin()};
}


__host__ __device__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::begin_projections() const
{
    return device_projections_.cbegin();
}


__host__ __device__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::end_projections()
{
    return ProjectionIterator{device_projections_.end()};
}


__host__ __device__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::end_projections() const
{
    return device_projections_.cend();
}

}  // namespace knp::backends::gpu::cuda
