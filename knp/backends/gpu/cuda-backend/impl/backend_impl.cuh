/**
 * @file backend_impl.cuh
 * @brief Class definition for CUDA GPU backend.
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

#pragma once

#include <knp/backends/gpu-cuda/backend.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/message_bus.h>
#include <knp/devices/gpu_cuda.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/config.hpp>
#include <boost/dll/alias.hpp>
#include <boost/mp11.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <cuda/std/variant>

#include "projection.cuh"
#include "population.cuh"
#include "cuda_bus/message_bus.cuh"
#include "cuda_lib/vector.cuh"


/**
 * @brief Namespace for single-threaded backend.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief The CUDABackend class is a definition of an interface to the CUDA GPU backend.
 */
class CUDABackendImpl
{
public:
    /**
     * @brief List of neuron types supported by the single-threaded CPU backend.
     */
    using SupportedNeurons = boost::mp11::mp_list<knp::neuron_traits::BLIFATNeuron>;

    /**
     * @brief List of synapse types supported by the single-threaded CPU backend.
     */
    using SupportedSynapses = boost::mp11::mp_list<knp::synapse_traits::DeltaSynapse>;

    /**
     * @brief List of supported population types based on neuron types specified in `SupportedNeurons`.
     */
    using SupportedPopulations = boost::mp11::mp_transform<CUDAPopulation, SupportedNeurons>;

    /**
     * @brief List of supported projection types based on synapse types specified in `SupportedSynapses`.
     */
    using SupportedProjections = boost::mp11::mp_transform<CUDAProjection, SupportedSynapses>;

    /**
     * @brief Population variant that contains any population type specified in `SupportedPopulations`.
     * @details `PopulationVariants` takes the value of `std::variant<PopulationType_1,..., PopulationType_n>`, where
     * `PopulationType_[1..n]` is the population type specified in `SupportedPopulations`. \n
     * For example, if `SupportedPopulations` contains BLIFATNeuron and IzhikevichNeuron types,
     * then `PopulationVariants = std::variant<BLIFATNeuron, IzhikevichNeuron>`. \n
     * `PopulationVariants` retains the same order of message types as defined in `SupportedPopulations`.
     * @see ALL_NEURONS.
     */
    using PopulationVariants = boost::mp11::mp_rename<SupportedPopulations, ::cuda::std::variant>;

    /**
     * @brief Projection variant that contains any projection type specified in `SupportedProjections`.
     * @details `ProjectionVariants` takes the value of `std::variant<ProjectionType_1,..., ProjectionType_n>`, where
     * `ProjectionType_[1..n]` is the projection type specified in `SupportedProjections`. \n
     * For example, if `SupportedProjections` contains DeltaSynapse and AdditiveSTDPSynapse types,
     * then `ProjectionVariants = std::variant<DeltaSynapse, AdditiveSTDPSynapse>`. \n
     * `ProjectionVariants` retains the same order of message types as defined in `SupportedProjections`.
     * @see ALL_SYNAPSES.
     */
    using ProjectionVariants = boost::mp11::mp_rename<SupportedProjections, ::cuda::std::variant>;

    /**
     * @brief Map used for message construction. It maps a message to its future output step.
     */
    using SynapticMessageQueue = std::unordered_map<uint64_t, core::messaging::SynapticImpactMessage>;

public:
    /**
     * @brief Type of population container.
     */
    using PopulationContainer = device_lib::CUDAVector<PopulationVariants>;

    /**
     * @brief Type of projection container.
     */
    using ProjectionContainer = device_lib::CUDAVector<ProjectionVariants>;

    /**
     * @brief Types of non-constant population iterators.
     */
    using PopulationIterator = typename PopulationContainer::iterator;

    /**
     * @brief Types of non-constant projection iterators.
     */
    using ProjectionIterator = typename ProjectionContainer::iterator;

    /**
     * @brief Types of constant population iterators.
     */
    using PopulationConstIterator = typename PopulationContainer::const_iterator;

    /**
     * @brief Types of constant projection iterators.
     */
    using ProjectionConstIterator = typename ProjectionContainer::const_iterator;

public:
    /**
     * @brief Constructor.
     * @param cpu_bus Bus to exchange backend with external world.
     */
    __host__ explicit CUDABackendImpl(knp::core::MessageEndpoint &message_endpoint) :
        device_message_bus_{message_endpoint} {}

    /**
     * @brief Destructor for  backend.
     */
    ~CUDABackendImpl() = default;

public:
    /**
     * @brief Add projections to backend.
     * @throw exception if the `projections` parameter contains unsupported projection types.
     * @param projections projections to add.
     */
    __host__ void load_projections(const knp::backends::gpu::CUDABackend::ProjectionContainer &projections);

    /**
     * @brief Add populations to backend.
     * @throw exception if the `populations` parameter contains unsupported population types.
     * @param populations populations to add.
     */
    __host__ void load_populations(const knp::backends::gpu::CUDABackend::PopulationContainer &populations);

public:
    /**
     * @brief Get an iterator pointing to the first element of the population loaded to backend.
     * @return population iterator.
     */
    __host__ __device__ PopulationIterator begin_populations();

    /**
     * @brief Get an iterator pointing to the first element of the population loaded to backend.
     * @return constant population iterator.
     */
    __host__ __device__ PopulationConstIterator begin_populations() const;
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * @return iterator.
     */
    __host__ __device__ PopulationIterator end_populations();
    /**
     * @brief Get a constant iterator pointing to the last element of the population.
     * @return iterator.
     */
    __host__ __device__ PopulationConstIterator end_populations() const;

    /**
     * @todo Make iterator which returns projections, but not a wrapper.
     */
    /**
     * @brief Get an iterator pointing to the first element of the projection loaded to backend.
     * @return projection iterator.
     */
    __host__ __device__ ProjectionIterator begin_projections();
    /**
     * @brief Get an iterator pointing to the first element of the projection loaded to backend.
     * @return constant projection iterator.
     */
    __host__ __device__ ProjectionConstIterator begin_projections() const;
    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * @return iterator.
     */
    __host__ __device__ ProjectionIterator end_projections();
    /**
     * @brief Get a constant iterator pointing to the last element of the projection.
     * @return iterator.
     */
    __host__ __device__ ProjectionConstIterator end_projections() const;

public:
    /**
     * @brief Remove projections with given UIDs from the backend.
     * @param uids UIDs of projections to remove.
     */
    __host__ __device__ void remove_projections(const std::vector<knp::core::UID> &uids) {}

    /**
     * @brief Remove populations with given UIDs from the backend.
     * @param uids UIDs of populations to remove.
     */
    __host__ __device__ void remove_populations(const std::vector<knp::core::UID> &uids) {}

public:
    /**
     * @brief Stop training by locking all projections.
     */
    __host__ void stop_learning()
    {
        for (auto &proj : device_projections_) ::cuda::std::visit([](auto &entity) { entity.lock_weights(); }, proj);
    }

    /**
     * @brief Resume training by unlocking all projections.
     */
    __host__ void start_learning()
    {
        /**
         * @todo Probably only need to use `start_learning` for some of projections: the ones that were locked with
         * `lock()`.
         */
        for (auto &proj : device_projections_) ::cuda::std::visit([](auto &entity) { entity.unlock_weights(); }, proj);
    }

    // __host__ uint64_t route_population_messages(uint64_t step);

    __host__ uint64_t route_projection_messages(uint64_t step);


    // [[nodiscard]] DataRanges get_network_data() const { return {}; }

public:
    __host__ void calculate_populations(std::uint64_t step);
    __host__ void calculate_projections(std::uint64_t step);
    __host__ knp::backends::gpu::cuda::CUDAMessageBus &get_message_bus() { return device_message_bus_; }

public:
    /**
     * @brief Calculate population of BLIFAT neurons.
     * @note Population will be changed during calculation.
     * @param population population to calculate.
     * @return copy of a spike message if population is emitting one.
     */
    static __device__ ::cuda::std::optional<knp::backends::gpu::cuda::SpikeMessage> calculate_population(
        CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
        std::uint64_t step_n);

    static __device__ ::cuda::std::optional<knp::backends::gpu::cuda::SpikeMessage> calculate_population(
        CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
        knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
        std::uint64_t step_n);


    /**
     * @brief Calculate projection of delta synapses.
     * @note Projection will be changed during calculation.
     * @param projection projection to calculate.
     * @param message_queue message queue to send to projection for calculation.
     */
    static __device__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
        std::uint64_t step_n);

    static __device__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
        std::uint64_t step_n);

    static __device__ void calculate_projection(
        CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
        std::uint64_t step_n);

    void init();

private:
    // cppcheck-suppress unusedStructMember
    PopulationContainer device_populations_;
    ProjectionContainer device_projections_;

    knp::backends::gpu::cuda::CUDAMessageBus device_message_bus_;
};


template <>
CUDABackendImpl::PopulationVariants gpu_extract<CUDABackendImpl::PopulationVariants>(
        const CUDABackendImpl::PopulationVariants *message);

template <>
void gpu_insert<CUDABackendImpl::PopulationVariants>(const CUDABackendImpl::PopulationVariants &cpu_source,
                                                     CUDABackendImpl::PopulationVariants *gpu_target);

template <>
CUDABackendImpl::ProjectionVariants gpu_extract<CUDABackendImpl::ProjectionVariants>(
        const CUDABackendImpl::ProjectionVariants *message);

template <>
void gpu_insert<CUDABackendImpl::ProjectionVariants>(const CUDABackendImpl::ProjectionVariants &cpu_source,
                                                     CUDABackendImpl::ProjectionVariants *gpu_target);

}  // namespace knp::backends::gpu::cuda
