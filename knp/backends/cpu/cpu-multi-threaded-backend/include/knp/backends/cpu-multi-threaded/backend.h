/**
 * @file backend.h
 * @brief Class definition for multi-threaded CPU backend.
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

#pragma once

#include <knp/backends/thread_pool/thread_pool.h>
#include <knp/core/backend.h>
#include <knp/core/impexp.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/devices/cpu.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/asio.hpp>
#include <boost/config.hpp>
#include <boost/dll/alias.hpp>
#include <boost/mp11.hpp>


/**
 * @brief Namespace for CPU backend executors.
 */
namespace knp::backends::cpu_executors
{
/**
 * @brief The ThreadPool class is an internal thread pool class used for task scheduling.
 */
class ThreadPool;
}  // namespace knp::backends::cpu_executors

/**
 * @brief Namespace for multi-threaded backend.
 */
namespace knp::backends::multi_threaded_cpu
{
/**
 * @brief Default size of a population part that is processed in a single thread.
 */
const size_t default_population_part_size = 1000;

/**
 * @brief Default size of a projection part that is processed in a single thread.
 */
const size_t default_projection_part_size = 1000;

/**
 * @brief The MultiThreadedCPUBackend class is a definition of an interface to the multi-threaded CPU backend.
 */
class KNP_DECLSPEC MultiThreadedCPUBackend : public knp::core::Backend
{
public:
    /**
     * @brief List of neuron types supported by the multi-threaded CPU backend.
     */
    using SupportedNeurons =
        boost::mp11::mp_list<knp::neuron_traits::BLIFATNeuron, knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;

    /**
     * @brief List of synapse types supported by the multi-threaded CPU backend.
     */
    using SupportedSynapses =
        boost::mp11::mp_list<knp::synapse_traits::DeltaSynapse, knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;

    /**
     * @brief List of supported population types based on neuron types specified in `SupportedNeurons`.
     */
    using SupportedPopulations = boost::mp11::mp_transform<knp::core::Population, SupportedNeurons>;

    /**
     * @brief List of supported projection types based on synapse types specified in `SupportedSynapses`.
     */
    using SupportedProjections = boost::mp11::mp_transform<knp::core::Projection, SupportedSynapses>;
    /**
     * @brief Population variant that contains any population type specified in `SupportedPopulations`.
     * @details `PopulationVariants` takes the value of `std::variant<PopulationType_1,..., PopulationType_n>`, where
     * `PopulationType_[1..n]` is the population type specified in `SupportedPopulations`. \n
     * For example, if `SupportedPopulations` contains BLIFATNeuron and IzhikevichNeuron types,
     * then `PopulationVariants = std::variant<BLIFATNeuron, IzhikevichNeuron>`. \n
     * `PopulationVariants` retains the same order of message types as defined in `SupportedPopulations`.
     * @see ALL_NEURONS.
     */
    using PopulationVariants = boost::mp11::mp_rename<SupportedPopulations, std::variant>;
    /**
     * @brief Projection variant that contains any projection type specified in `SupportedProjections`.
     * @details `ProjectionVariants` takes the value of `std::variant<ProjectionType_1,..., ProjectionType_n>`, where
     * `ProjectionType_[1..n]` is the projection type specified in `SupportedProjections`. \n
     * For example, if `SupportedProjections` contains DeltaSynapse and AdditiveSTDPSynapse types,
     * then `ProjectionVariants = std::variant<DeltaSynapse, AdditiveSTDPSynapse>`. \n
     * `ProjectionVariants` retains the same order of message types as defined in `SupportedProjections`.
     * @see ALL_SYNAPSES.
     */
    using ProjectionVariants = boost::mp11::mp_rename<SupportedProjections, std::variant>;

private:
    struct ProjectionWrapper
    {
        ProjectionVariants arg_;
        std::unordered_map<uint64_t, knp::core::messaging::SynapticImpactMessage> messages_;
    };

public:
    /**
     * @brief Type of population container.
     */
    using PopulationContainer = std::vector<PopulationVariants>;
    /**
     * @brief Type of projection container.
     */
    using ProjectionContainer = std::vector<ProjectionWrapper>;

    /**
     * @brief Types of non-constant population iterators.
     */
    using PopulationIterator = PopulationContainer::iterator;

    /**
     * @brief Types of non-constant projection iterators.
     */
    using ProjectionIterator = ProjectionContainer::iterator;

    /**
     * @brief Get a set of iterators for projections and populations.
     * @return `DataRanges` structure containing iterators.
     */
    [[nodiscard]] DataRanges get_network_data() const override;


    /**
     * @brief Types of constant population iterators.
     */
    using PopulationConstIterator = PopulationContainer::const_iterator;

    /**
     * @brief Types of constant projection iterators.
     */
    using ProjectionConstIterator = ProjectionContainer::const_iterator;

public:
    /**
     * @brief Default constructor for multi-threaded CPU backend.
     * @param thread_count number of threads.
     * @param population_part_size number of synapses that are calculated in a single thread.
     * @param projection_part_size number of neurons that are calculated in a single thread.
     * @note If `thread_count` equals `0`, then the number of threads is calculated automatically.
     */
    explicit MultiThreadedCPUBackend(
        size_t thread_count = 0, size_t population_part_size = default_population_part_size,
        size_t projection_part_size = default_projection_part_size);
    /**
     * @brief Destructor for multi-threaded CPU backend.
     * @note All threads are stopped and joined on destruction by an internal thread pool object.
     */
    ~MultiThreadedCPUBackend() override = default;

public:
    /**
     * @brief Create an object of the multi-threaded CPU backend.
     * @return shared pointer to backend object.
     */
    static std::shared_ptr<MultiThreadedCPUBackend> create();

public:
    /**
     * @brief Define if plasticity is supported.
     * @return `true` if plasticity is supported, `false` if plasticity is not supported.
     */
    [[nodiscard]] bool plasticity_supported() const override { return true; }
    /**
     * @brief Get type names of supported neurons.
     * @return vector of supported neuron type names.
     */
    [[nodiscard]] std::vector<std::string> get_supported_neurons() const override;
    /**
     * @brief Get type names of supported synapses.
     * @return vector of supported synapse type names.
     */
    [[nodiscard]] std::vector<std::string> get_supported_synapses() const override;
    /**
     * @brief Get indexes of supported projections.
     * @return vector of supported type indexes.
     */
    [[nodiscard]] std::vector<size_t> get_supported_projection_indexes() const override;
    /**
     * @brief Get indexes of supported populations.
     * @return vector of supported type indexes.
     */
    [[nodiscard]] std::vector<size_t> get_supported_population_indexes() const override;

public:
    /**
     * @brief Load populations to the backend.
     * @param populations vector of populations to load.
     */
    void load_populations(const std::vector<PopulationVariants> &populations);

    /**
     * @brief Load projections to the backend.
     * @param projections vector of projections to load.
     */
    void load_projections(const std::vector<ProjectionVariants> &projections);

    /**
     * @brief Add projections to backend.
     * @throw exception if the `projections` parameter contains unsupported projection types.
     * @param projections projections to add.
     */
    void load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections) override;

    /**
     * @brief Add populations to backend.
     * @throw exception if the `populations` parameter contains unsupported population types.
     * @param populations populations to add.
     */
    void load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations) override;

public:
    /**
     * @brief Get an iterator pointing to the first element of the population loaded to backend.
     * @return population iterator.
     */
    [[nodiscard]] PopulationIterator begin_populations();
    /**
     * @brief Get an iterator pointing to the first element of the population loaded to backend.
     * @return constant population iterator.
     */
    [[nodiscard]] PopulationConstIterator begin_populations() const;
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * @return iterator.
     */
    [[nodiscard]] PopulationIterator end_populations();
    /**
     * @brief Get a constant iterator pointing to the last element of the population.
     * @return iterator.
     */
    [[nodiscard]] PopulationConstIterator end_populations() const;
    /**
     * @brief Get an iterator pointing to the first element of the projection loaded to backend.
     * @return projection iterator.
     */
    [[nodiscard]] ProjectionIterator begin_projections();
    /**
     * @brief Get an iterator pointing to the first element of the projection loaded to backend.
     * @return constant projection iterator.
     */
    [[nodiscard]] ProjectionConstIterator begin_projections() const;
    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * @return iterator.
     */
    [[nodiscard]] ProjectionIterator end_projections();
    /**
     * @brief Get a constant iterator pointing to the last element of the projection.
     * @return iterator.
     */
    [[nodiscard]] ProjectionConstIterator end_projections() const;

public:
    /**
     * @brief Remove projections with given UIDs from the backend.
     * @param uids UIDs of projections to remove.
     */
    void remove_projections(const std::vector<knp::core::UID> &uids) override {}

    /**
     * @brief Remove populations with given UIDs from the backend.
     * @param uids UIDs of populations to remove.
     */
    void remove_populations(const std::vector<knp::core::UID> &uids) override {}

public:
    /**
     * @brief Get a list of devices supported by the backend.
     * @return list of devices.
     * @see Device.
     */
    [[nodiscard]] std::vector<std::unique_ptr<knp::core::Device>> get_devices() const override;

public:
    /**
     * @copydoc knp::core::Backend::_step()
     */
    void _step() override;

    /**
     * @brief Calculate all populations.
     */
    void calculate_populations();

    /**
     * @brief Calculate all projections.
     */
    void calculate_projections();

    /**
     * @brief Stop training by locking all projections.
     */
    void stop_learning() override
    {
        for (ProjectionWrapper &wrapper : projections_)
            std::visit([](auto &entity) { entity.lock_weights(); }, wrapper.arg_);
    }

    /**
     * @brief Resume training by unlocking all projections.
     */
    void start_learning() override
    {
        /**
         * @todo Probably only need to use `start_learning` for some of projections: the ones that were locked with
         * `lock()`.
         */
        for (ProjectionWrapper &wrapper : projections_)
            std::visit([](auto &entity) { entity.unlock_weights(); }, wrapper.arg_);
    }

protected:
    /**
     * @copydoc knp::core::Backend::_init()
     */
    void _init() override;

private:
    // Calculating pre-message neuron state, one thread per population_part_size_ neurons or less.
    void calculate_populations_pre_impact();
    // Processing messages, one thread per population, probably very hard to go deeper unless atomic neuron params.
    void calculate_populations_impact();
    // Do STDP logic for populations that support it. One thread per population.
    void do_STDP();
    // Calculating post input changes and outputs.
    std::vector<knp::core::messaging::SpikeMessage> calculate_populations_post_impact();
    PopulationContainer populations_;
    ProjectionContainer projections_;
    const size_t population_part_size_;
    const size_t projection_part_size_;
    std::unique_ptr<cpu_executors::ThreadPool> calc_pool_;
    std::mutex ep_mutex_;
};

}  // namespace knp::backends::multi_threaded_cpu
