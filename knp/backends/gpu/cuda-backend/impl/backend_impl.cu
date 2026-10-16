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

#include "cuda_lib/fast_error_check.cuh"
#include "cuda_lib/get_blocks_config.cuh"
#include "cuda_lib/printf.cuh"
#include "cuda_lib/register_all.cuh"
#include "cuda_lib/vector.cuh"
#include "cuda_bus/messaging.cuh"
#include <thrust/binary_search.h>
#include <thrust/sort.h>


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


template <>
PopulationVariants gpu_extract<PopulationVariants>(const PopulationVariants *);

template <>
void gpu_insert<PopulationVariants>(const PopulationVariants &, PopulationVariants *);

template <>
ProjectionVariants gpu_extract<ProjectionVariants>(const ProjectionVariants *);

template <>
void gpu_insert<ProjectionVariants>(const ProjectionVariants &, ProjectionVariants *);

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


__global__ void get_population_kernel(const PopulationVariants *var, int *type, const void **pop)
{
    get_kernel(var, type, pop);
}


__global__ void get_projection_kernel(const ProjectionVariants *var, int *type, const void **proj)
{
    get_kernel(var, type, proj);
}


template<>
void gpu_insert<PopulationVariants>(const PopulationVariants &cpu_source, PopulationVariants *gpu_target)
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


template<class T>
__global__ void get_uids_kernel(const T *data, size_t size, cuda::UID *result)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
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


template<class VectorData>
device_lib::CUDAVector<cuda::UID> get_uids_std(const std::vector<VectorData> &entities)
{
    device_lib::CUDAVector<cuda::UID> result;
    result.reserve(entities.size());
    for (size_t i = 0; i < entities.size(); ++i)
    {
        ::cuda::std::visit([&result](const auto &entity)
        {
            result.push_back(entity.uid_);
        }, entities[i]);
    }
    return result;
}


void CUDABackendImpl::calculate_projections(StepIndex step)
{
    // Calculate projections.
    device_lib::CUDAVector<cuda::UID> projection_uids = get_uids_std(device_projections_);

    if (!device_projections_.size()) return;

    std::vector<std::vector<device_lib::LongIndex>> projection_messages;
    projection_messages.reserve(device_projections_.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const std::vector<device_lib::LongIndex> message_ids
            = device_message_bus_.unload_messages<cuda::SpikeMessage>(projection_uids.copy_at(i));
        projection_messages.push_back(message_ids);
    }
    assert(device_projections_.size() == projection_messages.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        ::cuda::std::visit([this, &projection_messages, step, i](auto &projection)
        {
            calculate_projection(projection, device_message_bus_, projection_messages[i], step);
        }, device_projections_[i]);
    }
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
    CUDA_FAST_ERROR_CHECK("Starting to load, already an error: {}");
    device_projections_.clear();
    CUDA_FAST_ERROR_CHECK("Cleared device projections: {}");
    device_projections_.reserve(projections.size());
    CUDA_FAST_ERROR_CHECK("Reserving device projections: {}");

    for (const auto &projection : projections)
    {
        ::std::visit([this](auto &arg)
        {
            using CPUProjectionType = std::decay_t<decltype(arg)>;

            auto proj = CUDAProjection<typename CPUProjectionType::ProjectionSynapseType>{arg};
            SPDLOG_DEBUG("Pushing back a projection, size before: {}, pointer before: {}, capacity {}",
                    device_projections_.size(), reinterpret_cast<void *>(device_projections_.data()),
                    device_projections_.capacity());
            device_projections_.push_back(proj);
            CUDA_FAST_ERROR_CHECK("Pushed back {}");
            SPDLOG_DEBUG("Pushed back: size after: {}, pointer after: {}, capacity {}", device_projections_.size(),
                    reinterpret_cast<void *>(device_projections_.data()), device_projections_.capacity());
        }, projection);
    }

    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackendImpl::init()
{
    SPDLOG_DEBUG("Initializing CUDABackendImpl...");

    // knp::backends::cpu::init(projections_, get_message_endpoint());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const auto [pre_uid, post_uid, this_uid] = ::cuda::std::visit([](auto &proj)
        {
            return std::make_tuple(proj.presynaptic_uid_, proj.postsynaptic_uid_, proj.uid_);
        }, device_projections_[i]);

        if (!cuda::empty_uid(pre_uid)) this->device_message_bus_.subscribe_gpu<cuda::SpikeMessage>(this_uid, {pre_uid});

        if (!cuda::empty_uid(post_uid))
        {
            this->device_message_bus_.subscribe_gpu<cuda::SynapticImpactMessage>(post_uid, {this_uid});
        }
    }

    SPDLOG_DEBUG("Initialization finished.");
}


void CUDABackendImpl::calculate_populations(StepIndex step)
{
    // device_message_bus_.clear<SpikeMessage>();
    for (auto &population : device_populations_)
    {
        ::cuda::std::visit([this, step](auto &pop)
        {
            auto spikes = calculate_population(pop, device_message_bus_, step);
            if (!spikes.empty())
            {
                cuda::SpikeMessage message{MessageHeader{pop.uid_, step, false}, std::move(spikes)};
                SPDLOG_DEBUG("Population {} created a message with {} spikes", std::string(to_cpu_uid(pop.uid_)),
                             message.neuron_indexes_.size());
                device_message_bus_.send_message(std::move(message));
            }
            else
            {
                SPDLOG_DEBUG("No spike messages were sent from population {}", ::std::string(to_cpu_uid(pop.uid_)));
            }
        }, population);
    }
}


__host__ uint64_t CUDABackendImpl::route_projection_messages(StepIndex step)
{
    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
    device_lib::LongIndex sent_message_counter = 0;
    device_message_bus_.clear();

    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        ::cuda::std::visit([this, &sent_message_counter](auto &proj)
        {
            if (proj.message_buf_.impacts_.size())
            {
                device_message_bus_.send_message(std::move(proj.message_buf_));
                // cudaDeviceSynchronize();
                proj.message_buf_.impacts_.clear();
                ++sent_message_counter;
            }
        }, device_projections_[i]);
    }
    cudaDeviceSynchronize();
    SPDLOG_DEBUG("Projections sent {} messages", sent_message_counter);
    return sent_message_counter;
}


__host__ CUDABackendImpl::PopulationIterator CUDABackendImpl::begin_populations()
{
    return PopulationIterator{device_populations_.begin()};
}


__host__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::begin_populations() const
{
    return {device_populations_.cbegin()};
}


__host__ CUDABackendImpl::PopulationIterator CUDABackendImpl::end_populations()
{
    return PopulationIterator{device_populations_.end()};
}


__host__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::end_populations() const
{
    return device_populations_.cend();
}


__host__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::begin_projections()
{
    return ProjectionIterator{device_projections_.begin()};
}


__host__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::begin_projections() const
{
    return device_projections_.cbegin();
}


__host__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::end_projections()
{
    return ProjectionIterator{device_projections_.end()};
}


__host__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::end_projections() const
{
    return device_projections_.cend();
}


__global__ void get_spike_message_data(device_lib::CUDAVectorView<cuda::MessageVariant> all_messages,
               device_lib::LongIndex msg_index, device_lib::LongIndex *size, const SpikeIndex **data_pointer)
{
    constexpr size_t spike_message_index = boost::mp11::mp_find<cuda::MessageVariant, cuda::SpikeMessage>();
    auto &message_var = all_messages.data_[msg_index];
    if (message_var.index() != spike_message_index)
    {
        *size = 0;
        *data_pointer = nullptr;
        PRINTF_DEBUG("Pointer (no msg): %p, size: %lu\n", *data_pointer, *size);
        return;
    }
    *data_pointer = ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.data();
    *size = ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.size();
#ifdef DEBUG
    printf("Pointer: %p, size: %lu\n", *data_pointer, *size);
    for (size_t i = 0; i < *size; ++i)
    {
        printf("%u ",  ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.data()[i]);
    }
    printf("\n");
#endif // DEBUG
}

}   // namespace knp::backends::gpu::cuda


// TODO: Replace with a loop macro.
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::PopulationVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::ProjectionVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>::NeuronParameters);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
