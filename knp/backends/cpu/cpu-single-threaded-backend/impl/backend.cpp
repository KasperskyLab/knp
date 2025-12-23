/**
 * @file backend.cpp
 * @brief Single-threaded CPU backend class implementation.
 * @kaspersky_support Artiom N.
 * @date 21.02.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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


#include <knp/backends/cpu-library/altai_lif_population.h>
#include <knp/backends/cpu-library/blifat_population.h>
#include <knp/backends/cpu-library/delta_synapse_projection.h>
#include <knp/backends/cpu-library/init.h>
#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/devices/cpu.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>
#include <knp/synapse-traits/stdp_synaptic_resource_rule.h>

#include <spdlog/spdlog.h>

#include <vector>

#include <boost/mp11.hpp>

namespace knp::backends::single_threaded_cpu
{

SingleThreadedCPUBackend::SingleThreadedCPUBackend()
{
    SPDLOG_INFO("Single-threaded CPU backend instance created.");
}


std::shared_ptr<SingleThreadedCPUBackend> SingleThreadedCPUBackend::create()
{
    SPDLOG_DEBUG("Creating single-threaded CPU backend instance...");
    return std::make_shared<SingleThreadedCPUBackend>();
}


std::vector<std::string> SingleThreadedCPUBackend::get_supported_neurons() const
{
    return knp::meta::get_supported_type_names<knp::neuron_traits::AllNeurons, SupportedNeurons>(
        knp::neuron_traits::neurons_names);
}


std::vector<std::string> SingleThreadedCPUBackend::get_supported_synapses() const
{
    return knp::meta::get_supported_type_names<knp::synapse_traits::AllSynapses, SupportedSynapses>(
        knp::synapse_traits::synapses_names);
}


std::vector<size_t> SingleThreadedCPUBackend::get_supported_projection_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllProjections, SupportedProjections>();
}


std::vector<size_t> SingleThreadedCPUBackend::get_supported_population_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllPopulations, SupportedPopulations>();
}


template <typename AllVariants, typename SupportedVariants>
SupportedVariants convert_variant(const AllVariants &input)
{
    SupportedVariants result = std::visit([](auto &&arg) { return arg; }, input);
    return result;
}


void SingleThreadedCPUBackend::_step()
{
    SPDLOG_DEBUG("Starting step #{}...", get_step());
    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    // Calculate populations. This is the same as inference.
    for (auto &population : populations_)
    {
        std::visit(
            [this](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (
                    boost::mp11::mp_find<SupportedPopulations, T>{} == boost::mp11::mp_size<SupportedPopulations>{})
                {
                    static_assert(
                        knp::meta::always_false_v<T>,
                        "Population is not supported by the single-threaded CPU backend.");
                }
                auto message_opt = calculate_population(arg);
            },
            population);
    }

    // Continue inference.
    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    // Calculate projections.
    for (auto &projection : projections_)
    {
        std::visit(
            [this, &projection](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (
                    boost::mp11::mp_find<SupportedProjections, T>{} == boost::mp11::mp_size<SupportedProjections>{})
                {
                    static_assert(
                        knp::meta::always_false_v<T>,
                        "Projection is not supported by the single-threaded CPU backend.");
                }
                calculate_projection(arg, projection.messages_);
            },
            projection.arg_);
    }

    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    auto step = gad_step();
    // Need to suppress "Unused variable" warning.
    (void)step;
    SPDLOG_DEBUG("Step finished #{}.", step);
}


void SingleThreadedCPUBackend::load_populations(const std::vector<PopulationVariants> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    populations_.clear();
    populations_.reserve(populations.size());

    for (const auto &population : populations)
    {
        populations_.push_back(population);
    }
    SPDLOG_DEBUG("All populations loaded.");
}


void SingleThreadedCPUBackend::load_projections(const std::vector<ProjectionVariants> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    projections_.clear();
    projections_.reserve(projections.size());

    for (const auto &projection : projections)
    {
        projections_.push_back(ProjectionWrapper{projection});
    }

    SPDLOG_DEBUG("All projections loaded.");
}


void SingleThreadedCPUBackend::load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    knp::meta::load_from_container<SupportedProjections>(projections, projections_);
    SPDLOG_DEBUG("All projections loaded.");
}


void SingleThreadedCPUBackend::load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    knp::meta::load_from_container<SupportedPopulations>(populations, populations_);
    SPDLOG_DEBUG("All populations loaded.");
}


std::vector<std::unique_ptr<knp::core::Device>> SingleThreadedCPUBackend::get_devices() const
{
    std::vector<std::unique_ptr<knp::core::Device>> result;
    auto &&processors{knp::devices::cpu::list_processors()};

    result.reserve(processors.size());

    for (auto &&cpu : processors)
    {
        SPDLOG_DEBUG("Device CPU \"{}\".", cpu.get_name());
        result.push_back(std::make_unique<knp::devices::cpu::CPU>(std::move(cpu)));
    }

    SPDLOG_DEBUG("CPU count = {}.", result.size());
    return result;
}


void SingleThreadedCPUBackend::_init()
{
    SPDLOG_DEBUG("Initializing single-threaded CPU backend...");

    knp::backends::cpu::init(projections_, get_message_endpoint());

    SPDLOG_DEBUG("Initialization finished.");
}


std::optional<core::messaging::SpikeMessage> SingleThreadedCPUBackend::calculate_population(
    core::Population<knp::neuron_traits::BLIFATNeuron> &population)
{
    SPDLOG_TRACE("Calculate BLIFAT population {}.", std::string(population.get_uid()));
    return knp::backends::cpu::calculate_blifat_population(population, get_message_endpoint(), get_step());
}


std::optional<core::messaging::SpikeMessage> SingleThreadedCPUBackend::calculate_population(
    core::Population<neuron_traits::AltAILIF> &population)
{
    SPDLOG_TRACE("Calculate AltAI-LIF population {}.", std::string(population.get_uid()));
    return knp::backends::cpu::calculate_lif_population<neuron_traits::AltAILIF>(
        population, get_message_endpoint(), get_step());
}


std::optional<core::messaging::SpikeMessage> SingleThreadedCPUBackend::calculate_population(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population)
{
    SPDLOG_TRACE("Calculate resource-based STDP-compatible BLIFAT population {}.", std::string(population.get_uid()));
    return knp::backends::cpu::calculate_resource_stdp_population<
        neuron_traits::BLIFATNeuron, synapse_traits::DeltaSynapse, ProjectionContainer>(
        population, projections_, get_message_endpoint(), get_step());
}


void SingleThreadedCPUBackend::calculate_projection(
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> &projection, SynapticMessageQueue &message_queue)
{
    SPDLOG_TRACE("Calculate delta synapse projection {}.", std::string(projection.get_uid()));
    knp::backends::cpu::calculate_delta_synapse_projection(
        projection, get_message_endpoint(), message_queue, get_step());
}


void SingleThreadedCPUBackend::calculate_projection(
    knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
    SynapticMessageQueue &message_queue)
{
    SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
    knp::backends::cpu::calculate_delta_synapse_projection(
        projection, get_message_endpoint(), message_queue, get_step());
}


void SingleThreadedCPUBackend::calculate_projection(
    knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
    SynapticMessageQueue &message_queue)
{
    SPDLOG_TRACE("Calculate STDPSynapticResource synapse projection {}.", std::string(projection.get_uid()));
    knp::backends::cpu::calculate_delta_synapse_projection(
        projection, get_message_endpoint(), message_queue, get_step());
}


SingleThreadedCPUBackend::PopulationIterator SingleThreadedCPUBackend::begin_populations()
{
    return PopulationIterator{populations_.begin()};
}


SingleThreadedCPUBackend::PopulationConstIterator SingleThreadedCPUBackend::begin_populations() const
{
    return {populations_.cbegin()};
}


SingleThreadedCPUBackend::PopulationIterator SingleThreadedCPUBackend::end_populations()
{
    return PopulationIterator{populations_.end()};
}


SingleThreadedCPUBackend::PopulationConstIterator SingleThreadedCPUBackend::end_populations() const
{
    return populations_.cend();
}


SingleThreadedCPUBackend::ProjectionIterator SingleThreadedCPUBackend::begin_projections()
{
    return ProjectionIterator{projections_.begin()};
}


SingleThreadedCPUBackend::ProjectionConstIterator SingleThreadedCPUBackend::begin_projections() const
{
    return projections_.cbegin();
}


SingleThreadedCPUBackend::ProjectionIterator SingleThreadedCPUBackend::end_projections()
{
    return ProjectionIterator{projections_.end()};
}


SingleThreadedCPUBackend::ProjectionConstIterator SingleThreadedCPUBackend::end_projections() const
{
    return projections_.cend();
}


BOOST_DLL_ALIAS(knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::create, create_knp_backend)

}  // namespace knp::backends::single_threaded_cpu
