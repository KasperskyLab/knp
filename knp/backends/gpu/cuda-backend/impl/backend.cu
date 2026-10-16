/**
 * @file backend.cu
 * @brief CUDABackend backend class implementation.
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

#include "backend_impl.cuh"
#include "cuda_bus/message_bus.cuh"
#include "cuda_lib/fast_error_check.cuh"
#include "cuda_lib/register_all.cuh"

#include <spdlog/spdlog.h>

#include <knp/backends/gpu-cuda/backend.h>
#include <knp/devices/gpu_cuda.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <limits>
#include <vector>

#include <boost/mp11.hpp>


namespace knp::backends::gpu
{

CUDABackend::CUDABackend() : impl_(std::make_unique<cuda::CUDABackendImpl>(get_message_endpoint()))
{
    SPDLOG_INFO("CUDA backend instance created.");
}


CUDABackend::~CUDABackend()
{
}


std::shared_ptr<CUDABackend> CUDABackend::create()
{
    SPDLOG_DEBUG("Creating CUDA backend instance...");
    return std::make_shared<CUDABackend>();
}


std::vector<std::string> CUDABackend::get_supported_neurons() const
{
    return knp::meta::get_supported_type_names<knp::neuron_traits::AllNeurons, SupportedNeurons>(
        knp::neuron_traits::neurons_names);
}


std::vector<std::string> CUDABackend::get_supported_synapses() const
{
    return knp::meta::get_supported_type_names<knp::synapse_traits::AllSynapses, SupportedSynapses>(
        knp::synapse_traits::synapses_names);
}


std::vector<size_t> CUDABackend::get_supported_projection_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllProjections, SupportedProjections>();
}


std::vector<size_t> CUDABackend::get_supported_population_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllPopulations, SupportedPopulations>();
}


template <typename AllVariants, typename SupportedVariants>
SupportedVariants convert_variant(const AllVariants &input)
{
    SupportedVariants result = std::visit([](auto &&arg) { return arg; }, input);
    return result;
}


template <class ProjectionType>
constexpr bool is_forcing()
{
    return false;
}


template <>
constexpr bool is_forcing<knp::core::Projection<synapse_traits::DeltaSynapse>>()
{
    return true;
}


void CUDABackend::_step()
{
    auto step = get_step();
    cuda::CUDAMessageBus &bus = impl_->get_message_bus();
    SPDLOG_DEBUG("Starting step #{}...", step);
    if(!get_step())
    {
        bus.sync_with_host();
    }
    SPDLOG_DEBUG("Message bus 1 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    bus.send_messages_to_host<cuda::SpikeMessage>(step);
    bus.send_messages_to_host<cuda::SynapticImpactMessage>(step);
    SPDLOG_DEBUG("Message bus 2 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    get_message_bus().route_messages();
    SPDLOG_DEBUG("Message bus 3 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    bus.receive_messages_from_host();
    SPDLOG_DEBUG("Message bus 4 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());

    // Calculate populations. This is the same as inference.
    impl_->calculate_populations(step);
    SPDLOG_DEBUG("Message bus 5 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    // impl_->route_population_messages(step);  // this is a part of calculate_populations
    get_message_bus().route_messages();
    SPDLOG_DEBUG("Message bus 6 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    //
    // Calculate projections.
    impl_->calculate_projections(step);
    SPDLOG_DEBUG("Message bus 7 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    bus.send_messages_to_host<cuda::SynapticImpactMessage>(step);
    bus.send_messages_to_host<cuda::SpikeMessage>(step);
    SPDLOG_DEBUG("Message bus 8 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    get_message_bus().route_messages();
    SPDLOG_DEBUG("Message bus 9 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    SPDLOG_DEBUG("Step {}", step);
    impl_->route_projection_messages(step);
    SPDLOG_DEBUG("Message bus 10 spikes {} impacts {}", bus.get_num_messages<cuda::SpikeMessage>(),
                 bus.get_num_messages<cuda::SynapticImpactMessage>());
    SPDLOG_DEBUG("Step finished #{}.", get_step());

    step = gad_step();
    // Need to suppress "Unused variable" warning.
    (void)step;
}


void CUDABackend::stop_learning()
{
    SPDLOG_DEBUG("Stopping learning");
}


void CUDABackend::start_learning()
{
    SPDLOG_DEBUG("Starting learning");
}


void CUDABackend::load_populations(const std::vector<PopulationVariants> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    CUDA_FAST_ERROR_CHECK("Loading populations already with an error: {}");
    impl_->load_populations(populations);
    CUDA_FAST_ERROR_CHECK("An error occured during population loading: {}");
    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackend::load_projections(const std::vector<ProjectionVariants> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());

    //    projections_ = projections;
    CUDA_FAST_ERROR_CHECK("Running projections with error: {}");
    impl_->load_projections(projections);
    CUDA_FAST_ERROR_CHECK("Loaded projections with error: {}");
    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackend::load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    CUDA_FAST_ERROR_CHECK("1 {}");
    knp::meta::load_from_container<SupportedProjections>(projections, projections_);
    CUDA_FAST_ERROR_CHECK("2 {}");
    load_projections(projections_);
    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackend::load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    CUDA_FAST_ERROR_CHECK("2 {}");
    knp::meta::load_from_container<SupportedPopulations>(populations, populations_);
    CUDA_FAST_ERROR_CHECK("22 {}");
    load_populations(populations_);
    CUDA_FAST_ERROR_CHECK("222 {}");
    SPDLOG_DEBUG("All populations loaded.");
}


std::vector<std::unique_ptr<knp::core::Device>> CUDABackend::get_devices() const
{
    std::vector<std::unique_ptr<knp::core::Device>> result;
    auto &&processors{knp::devices::gpu::list_cuda_processors()};
    result.reserve(processors.size());
    for (auto &&gpu : processors)
    {
        SPDLOG_DEBUG("GPU \"{}\".", gpu.get_name());
        result.push_back(std::make_unique<knp::devices::gpu::CUDA>(std::move(gpu)));
    }

    SPDLOG_DEBUG("NVIDIA GPU count = {}.", result.size());
    return result;
}


void CUDABackend::select_devices(const std::set<knp::core::UID> &uids)
{
    Backend::select_devices(uids);

    if (get_current_devices().size() > 1)
    {
        SPDLOG_ERROR("In the current implementation only one GPU can be selected.");
        throw std::runtime_error("Only one GPU device can be selected");
    }

    auto cuda_device_ptr = dynamic_cast<knp::devices::gpu::CUDA*>(get_current_devices()[0].get());
    cudaSetDevice(cuda_device_ptr->get_socket_number());
}


void CUDABackend::select_device(std::unique_ptr<knp::core::Device> &&device)
{
    Backend::select_device(std::move(device));
    auto cuda_device_ptr = dynamic_cast<knp::devices::gpu::CUDA*>(get_devices()[0].get());

    cudaSetDevice(cuda_device_ptr->get_socket_number());
}


void CUDABackend::_init()
{
    SPDLOG_DEBUG("Initializing CUDABackend...");

    impl_->init();

    SPDLOG_DEBUG("Initialization finished.");
}


CUDABackend::PopulationIterator CUDABackend::begin_populations()
{
    return PopulationIterator{populations_.begin()};
}


CUDABackend::PopulationConstIterator CUDABackend::begin_populations() const
{
    return {populations_.cbegin()};
}


CUDABackend::PopulationIterator CUDABackend::end_populations()
{
    return PopulationIterator{populations_.end()};
}


CUDABackend::PopulationConstIterator CUDABackend::end_populations() const
{
    return populations_.cend();
}


CUDABackend::ProjectionIterator CUDABackend::begin_projections()
{
    return ProjectionIterator{projections_.begin()};
}


CUDABackend::ProjectionConstIterator CUDABackend::begin_projections() const
{
    return projections_.cbegin();
}


CUDABackend::ProjectionIterator CUDABackend::end_projections()
{
    return ProjectionIterator{projections_.end()};
}


CUDABackend::ProjectionConstIterator CUDABackend::end_projections() const
{
    return projections_.cend();
}


BOOST_DLL_ALIAS(knp::backends::gpu::CUDABackend::create, create_knp_backend)

}  // namespace knp::backends::gpu


// TODO: Probably wrap it into a recursive macro.
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::PopulationVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::ProjectionVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>::NeuronParameters);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
