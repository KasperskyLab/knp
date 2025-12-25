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
    SPDLOG_DEBUG("Starting step #{}...", step);
    if(!get_step()) impl_->get_message_bus().sync_with_host();
    SPDLOG_DEBUG("Message bus 1 {}", impl_->get_message_bus().get_num_messages());
    impl_->get_message_bus().send_messages_to_host(step);
    SPDLOG_DEBUG("Message bus 2 {}", impl_->get_message_bus().get_num_messages());
    get_message_bus().route_messages();
    SPDLOG_DEBUG("Message bus 3 {}", impl_->get_message_bus().get_num_messages());
    impl_->get_message_bus().receive_messages_from_host();
    SPDLOG_DEBUG("Message bus 4 {}", impl_->get_message_bus().get_num_messages());


    // Calculate populations. This is the same as inference.
    impl_->calculate_populations(step);
    SPDLOG_DEBUG("Message bus 5 {}", impl_->get_message_bus().get_num_messages());
    // impl_->route_population_messages(step);  // this is a part of calculate_populations
    get_message_bus().route_messages();
    SPDLOG_DEBUG("Message bus 6 {}", impl_->get_message_bus().get_num_messages());
    //
    // Calculate projections.
    impl_->calculate_projections(step);
    SPDLOG_DEBUG("Message bus 7 {}", impl_->get_message_bus().get_num_messages());
    impl_->get_message_bus().send_messages_to_host(step);
    SPDLOG_DEBUG("Message bus 8 {}", impl_->get_message_bus().get_num_messages());
    get_message_bus().route_messages();
    SPDLOG_DEBUG("Message bus 9 {}", impl_->get_message_bus().get_num_messages());
    impl_->route_projection_messages(step);
    SPDLOG_DEBUG("Message bus 10 {}", impl_->get_message_bus().get_num_messages());
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

    impl_->load_populations(populations);

    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackend::load_projections(const std::vector<ProjectionVariants> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());

    //    projections_ = projections;
    impl_->load_projections(projections);

    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackend::load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    knp::meta::load_from_container<SupportedProjections>(projections, projections_);
    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackend::load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    knp::meta::load_from_container<SupportedPopulations>(populations, populations_);
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
