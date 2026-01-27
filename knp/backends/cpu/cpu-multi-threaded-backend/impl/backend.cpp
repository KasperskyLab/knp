/**
 * @file backend.cpp
 * @brief Multi-threaded CPU backend class implementation.
 * @kaspersky_support Artiom N.
 * @date 21.06.2023
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

#include <knp/backends/cpu-library/init.h>
#include <knp/backends/cpu-library/populations.h>
#include <knp/backends/cpu-library/projections.h>
#include <knp/backends/cpu-multi-threaded/backend.h>
#include <knp/backends/thread_pool/thread_pool.h>
#include <knp/devices/cpu.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <spdlog/spdlog.h>

#include <functional>
#include <unordered_map>
#include <vector>

#include <boost/mp11.hpp>

namespace knp::backends::multi_threaded_cpu
{

MultiThreadedCPUBackend::MultiThreadedCPUBackend(
    size_t thread_count, size_t population_part_size, size_t projection_part_size)
    : population_part_size_(population_part_size),
      projection_part_size_(projection_part_size),
      calc_pool_(std::make_unique<cpu_executors::ThreadPool>(
          thread_count ? thread_count : std::thread::hardware_concurrency()))
{
    SPDLOG_INFO(
        "Multi-threaded CPU backend instance created, thread count = {}.",
        thread_count ? thread_count : std::thread::hardware_concurrency());
}


std::shared_ptr<MultiThreadedCPUBackend> MultiThreadedCPUBackend::create()
{
    SPDLOG_DEBUG("Creating multi-threaded CPU backend instance...");
    return std::make_shared<MultiThreadedCPUBackend>();
}


std::vector<std::string> MultiThreadedCPUBackend::get_supported_neurons() const
{
    return knp::meta::get_supported_type_names<knp::neuron_traits::AllNeurons, SupportedNeurons>(
        knp::neuron_traits::neurons_names);
}


std::vector<std::string> MultiThreadedCPUBackend::get_supported_synapses() const
{
    return knp::meta::get_supported_type_names<knp::synapse_traits::AllSynapses, SupportedSynapses>(
        knp::synapse_traits::synapses_names);
}


void MultiThreadedCPUBackend::calculate_populations_pre_impact()
{
    for (auto &population : populations_)
    {
        auto pop_size = std::visit([](auto &pop) { return pop.size(); }, population);
        for (size_t neuron_index = 0; neuron_index < pop_size; neuron_index += population_part_size_)
        {
            std::visit(
                [this, neuron_index](auto &pop)
                {
                    using T = std::decay_t<decltype(pop)>;
                    uint64_t part_end = std::min<uint64_t>(neuron_index + population_part_size_, pop.size());
                    calc_pool_->post(
                        cpu::populations::calculate_pre_impact_population_state<typename T::PopulationNeuronType>,
                        std::ref(pop), neuron_index, part_end);
                },
                population);
        }
    }
    // Wait for all threads to finish their work.
    calc_pool_->join();
}


void MultiThreadedCPUBackend::calculate_populations_impact()
{
    for (auto &population : populations_)
    {
        auto uid = std::visit([](auto &population) { return population.get_uid(); }, population);
        auto messages = get_message_endpoint().unload_messages<knp::core::messaging::SynapticImpactMessage>(uid);
        std::visit(
            [this, &messages](auto &pop)
            {
                using T = std::decay_t<decltype(pop)>;
                calc_pool_->post(
                    cpu::populations::impact_population<typename T::PopulationNeuronType>, std::ref(pop),
                    std::move(messages));
            },
            population);
    }
    calc_pool_->join();
}


std::vector<knp::core::messaging::SpikeMessage> MultiThreadedCPUBackend::calculate_populations_post_impact()
{
    std::vector<knp::core::messaging::SpikeMessage> spike_container(populations_.size());
    for (size_t pop_index = 0; pop_index < populations_.size(); ++pop_index)
    {
        auto &population = populations_[pop_index];
        auto &message = spike_container[pop_index];
        message.header_.send_time_ = get_step();
        message.header_.sender_uid_ = std::visit([](auto &population) { return population.get_uid(); }, population);

        const size_t population_size = std::visit([](auto &population) { return population.size(); }, population);
        for (size_t neuron_index = 0; neuron_index < population_size; neuron_index += population_part_size_)
        {
            std::visit(
                [this, &message, neuron_index](auto &pop)
                {
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4267)
#endif
                    using T = std::decay_t<decltype(pop)>;
                    auto call_calculate_post_input_state = [](T &pop_ref,
                                                              knp::core::messaging::SpikeMessage &message_ref,
                                                              size_t start, size_t end, std::mutex &mutex_ref)
                    {
                        uint64_t part_end = std::min<uint64_t>(start + end, pop_ref.size());
                        knp::core::messaging::SpikeMessage buffer_message;
                        knp::backends::cpu::populations::calculate_post_impact_population_state(
                            pop_ref, buffer_message, start, part_end);
                        const std::lock_guard lock(mutex_ref);
                        message_ref.neuron_indexes_.insert(
                            message_ref.neuron_indexes_.end(), buffer_message.neuron_indexes_.begin(),
                            buffer_message.neuron_indexes_.end());
                    };
                    calc_pool_->post(
                        call_calculate_post_input_state, std::ref(pop), std::ref(message), neuron_index,
                        population_part_size_, std::ref(ep_mutex_));
                },
                population);
        }
    }
    calc_pool_->join();
    for (size_t pop_id = 0; pop_id < populations_.size(); ++pop_id)
    {
        auto &message = spike_container[pop_id];
        std::visit(
            [this, &message](auto &pop)
            {
                using T = std::decay_t<decltype(pop)>;
                auto call_finalize = [](T &pop_ref, knp::core::messaging::SpikeMessage &message_ref,
                                        ProjectionContainer &proj_ref, knp::core::Step step)
                {
                    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
                    std::vector<std::reference_wrapper<knp::core::Projection<SynapseType>>> working_projections;
                    constexpr uint64_t type_index = boost::mp11::mp_find<
                        backends::multi_threaded_cpu::MultiThreadedCPUBackend::SupportedSynapses, SynapseType>();

                    for (auto &projection : proj_ref)
                    {
                        if (projection.arg_.index() != type_index) continue;

                        auto &actual_proj = std::get<type_index>(projection.arg_);
                        if (actual_proj.is_locked()) continue;

                        if (actual_proj.get_postsynaptic() == pop_ref.get_uid())
                            working_projections.push_back(actual_proj);
                    }

                    knp::backends::cpu::populations::train_population(pop_ref, working_projections, message_ref, step);
                };
                calc_pool_->post(call_finalize, std::ref(pop), std::ref(message), std::ref(projections_), get_step());
            },
            populations_[pop_id]);
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
    }
    calc_pool_->join();
    return spike_container;
}


void MultiThreadedCPUBackend::calculate_populations()
{
    SPDLOG_DEBUG("Calculating populations...");
    calculate_populations_pre_impact();

    calculate_populations_impact();

    auto spike_messages = calculate_populations_post_impact();

    // Sending non-empty messages.
    for (const auto &message : spike_messages)
    {
        if (message.neuron_indexes_.empty())
        {
            continue;
        }
        get_message_endpoint().send_message(message);
    }
}

template <class ProjectionWrapper>
void send_message(ProjectionWrapper &projection, core::MessageEndpoint &endpoint, uint64_t step)
{
    auto &msg_queue = projection.messages_;
    auto msg_iter = msg_queue.find(step);
    if (msg_iter != msg_queue.end())
    {
        endpoint.send_message(std::move(msg_iter->second));
        msg_queue.erase(msg_iter);
    }
}

inline std::unordered_map<uint64_t, size_t> convert_spikes(const core::messaging::SpikeMessage &message)
{
    std::unordered_map<knp::core::Step, size_t> result;
    for (auto neuron_idx : message.neuron_indexes_)
    {
        auto iter = result.find(neuron_idx);
        if (result.end() == iter)
            result.insert({neuron_idx, 1});
        else
            ++(iter->second);
    }
    return result;
}


void MultiThreadedCPUBackend::calculate_projections()
{
    SPDLOG_DEBUG("Calculating projections...");
    std::vector<std::unordered_map<uint64_t, size_t>> converted_message_buffer;
    converted_message_buffer.reserve(projections_.size());

    for (auto &projection : projections_)
    {
        auto uid = std::visit([](auto &proj) { return proj.get_uid(); }, projection.arg_);
        auto msg_buf = get_message_endpoint().unload_messages<knp::core::messaging::SpikeMessage>(uid);
        // We might want to add some preliminary function before, even if delta projection doesn't require it.
        if (msg_buf.empty())
        {
            continue;
        }

        // Looping over synapses.
        converted_message_buffer.emplace_back(convert_spikes(msg_buf[0]));
        const auto proj_size = std::visit([](const auto &proj) { return proj.size(); }, projection.arg_);
        for (size_t synapse_index = 0; synapse_index < proj_size; synapse_index += projection_part_size_)
        {
            std::visit(
                [this, synapse_index, &converted_message_buffer, &projection](auto &proj)
                {
                    using T = std::decay_t<decltype(proj)>;
                    calc_pool_->post(
                        cpu::projections::calculate_projection_multithreaded<typename T::ProjectionSynapseType>,
                        std::ref(proj), std::ref(converted_message_buffer.back()), std::ref(projection.messages_),
                        get_step(), synapse_index, projection_part_size_, std::ref(ep_mutex_));
                },
                projection.arg_);
        }
    }
    calc_pool_->join();
    // Sending messages. It might be possible to parallelize this as well if we use more than one endpoint.
    for (auto &projection : projections_)
    {
        auto &msg_queue = projection.messages_;
        auto msg_iter = msg_queue.find(get_step());
        if (msg_iter != msg_queue.end())
        {
            get_message_endpoint().send_message(std::move(msg_iter->second));
            msg_queue.erase(msg_iter);
        }
    }
}


std::vector<size_t> MultiThreadedCPUBackend::get_supported_projection_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllProjections, SupportedProjections>();
}


std::vector<size_t> MultiThreadedCPUBackend::get_supported_population_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllPopulations, SupportedPopulations>();
}


void MultiThreadedCPUBackend::_step()
{
    SPDLOG_DEBUG("Starting step #{}...", get_step());
    calculate_populations();
    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    calculate_projections();
    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    auto step = gad_step();
    // Need to suppress "Unused variable" warning.
    (void)step;
    SPDLOG_DEBUG("Step #{} finished.", step);
}


void MultiThreadedCPUBackend::load_populations(const std::vector<PopulationVariants> &populations)
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


void MultiThreadedCPUBackend::load_projections(const std::vector<ProjectionVariants> &projections)
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


void MultiThreadedCPUBackend::load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    knp::meta::load_from_container<SupportedProjections>(projections, projections_);
    SPDLOG_DEBUG("All projections loaded.");
}


void MultiThreadedCPUBackend::load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    knp::meta::load_from_container<SupportedPopulations>(populations, populations_);
    SPDLOG_DEBUG("All populations loaded.");
}


std::vector<std::unique_ptr<knp::core::Device>> MultiThreadedCPUBackend::get_devices() const
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


void MultiThreadedCPUBackend::_init()
{
    SPDLOG_DEBUG("Initializing multi-threaded CPU backend...");

    knp::backends::cpu::init(projections_, get_message_endpoint());

    SPDLOG_DEBUG("Initialization finished.");
}


MultiThreadedCPUBackend::PopulationIterator MultiThreadedCPUBackend::begin_populations()
{
    return populations_.begin();
}


MultiThreadedCPUBackend::PopulationConstIterator MultiThreadedCPUBackend::begin_populations() const
{
    return populations_.cbegin();
}


MultiThreadedCPUBackend::PopulationIterator MultiThreadedCPUBackend::end_populations()
{
    return populations_.end();
}


MultiThreadedCPUBackend::PopulationConstIterator MultiThreadedCPUBackend::end_populations() const
{
    return populations_.cend();
}


MultiThreadedCPUBackend::ProjectionIterator MultiThreadedCPUBackend::begin_projections()
{
    return projections_.begin();
}


MultiThreadedCPUBackend::ProjectionConstIterator MultiThreadedCPUBackend::begin_projections() const
{
    return projections_.cbegin();
}


MultiThreadedCPUBackend::ProjectionIterator MultiThreadedCPUBackend::end_projections()
{
    return projections_.end();
}


MultiThreadedCPUBackend::ProjectionConstIterator MultiThreadedCPUBackend::end_projections() const
{
    return projections_.cend();
}


BOOST_DLL_ALIAS(knp::backends::multi_threaded_cpu::MultiThreadedCPUBackend::create, create_knp_backend)
}  // namespace knp::backends::multi_threaded_cpu
