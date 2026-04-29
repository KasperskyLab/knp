/**
 * @file network.h
 * @brief Network interface.
 * @kaspersky_support Artiom N.
 * @date 22.03.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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

#include <knp/core/backend.h>
#include <knp/core/core.h>
#include <knp/core/impexp.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/coordinates/generator.h>
#include <knp/framework/projection/connectors.h>
#include <knp/framework/projection/synapse_parameters_generators.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief The Network class is a definition of a neural network that contains populations and projections.
 */
class KNP_DECLSPEC Network
{
public:
    /**
     * @brief List of population types based on neuron types specified in `knp::neuron_traits::AllNeurons`.
     * 
     * @details `AllPopulations` takes the value of `Population<NeuronType_1>, Population<NeuronType_2>, ...,
     * Population<NeuronType_n>`, where `NeuronType_[1..n]` is the neuron type specified in
     * `knp::neuron_traits::AllNeurons`. \n For example, if `knp::neuron_traits::AllNeurons` contains BLIFATNeuron and
     * IzhikevichNeuron types, then `AllPopulations` = `Population<BLIFATNeuron>, Population<IzhikevichNeuron>`.
     */
    using AllPopulations = boost::mp11::mp_transform<knp::core::Population, knp::neuron_traits::AllNeurons>;

    /**
     * @brief List of projection types based on synapse types specified in `knp::synapse_traits::AllSynapses`.
     * 
     * @details `AllProjections` takes the value of `Projection<SynapseType_1>, Projection<SynapseType_2>, ...,
     * Projection<SynapseType_n>`, where `SynapseType_[1..n]` is the synapse type specified in
     * `knp::synapse_traits::AllSynapses`. \n For example, if `knp::synapse_traits::AllSynapses` contains DeltaSynapse
     * and AdditiveSTDPSynapse types, then `AllProjections` = `Population<DeltaSynapse>,
     * Population<AdditiveSTDPSynapse>`.
     */
    using AllProjections = boost::mp11::mp_transform<knp::core::Projection, knp::synapse_traits::AllSynapses>;

    /**
     * @brief Population variant that contains any population type specified in `AllPopulations`.
     * 
     * @details `AllPopulationVariants` takes the value of `std::variant<PopulationType_1,..., PopulationType_n>`, where
     * `PopulationType_[1..n]` is the population type specified in `AllPopulations`. \n For example, if `AllPopulations`
     * contains BLIFATNeuron and IzhikevichNeuron types, then `AllPopulationVariants = std::variant<BLIFATNeuron,
     * IzhikevichNeuron>`. \n `AllPopulationVariants` retains the same order of message types as defined in
     * `AllPopulations`.
     * 
     * @see ALL_NEURONS.
     */
    using AllPopulationVariants = boost::mp11::mp_rename<AllPopulations, std::variant>;
    /**
     * @brief Projection variant that contains any projection type specified in `AllProjections`.
     * 
     * @details `AllProjectionVariants` takes the value of `std::variant<ProjectionType_1,..., ProjectionType_n>`, where
     * `ProjectionType_[1..n]` is the projection type specified in `AllProjections`. \n For example, if `AllProjections`
     * contains DeltaSynapse and AdditiveSTDPSynapse types, then `AllProjectionVariants = std::variant<DeltaSynapse,
     * AdditiveSTDPSynapse>`. \n `AllProjectionVariants` retains the same order of message types as defined in
     * `AllProjections`.
     * 
     * @see ALL_SYNAPSES.
     */
    using AllProjectionVariants = boost::mp11::mp_rename<AllProjections, std::variant>;

public:
    /**
     * @brief Type of population container.
     */
    using PopulationContainer = std::vector<core::AllPopulationsVariant>;
    /**
     * @brief Type of projection container.
     */
    using ProjectionContainer = std::vector<core::AllProjectionsVariant>;

    /**
     * @brief Types of population iterators.
     */
    using PopulationIterator = PopulationContainer::iterator;
    /**
     * @brief Types of constant population iterators.
     */
    using PopulationConstIterator = PopulationContainer::const_iterator;

    /**
     * @brief Types of projection iterators.
     */
    using ProjectionIterator = ProjectionContainer::iterator;
    /**
     * @brief Types of constant projection iterators.
     */
    using ProjectionConstIterator = ProjectionContainer::const_iterator;

public:
    /**
     * @brief Default network constructor.
     */
    Network() = default;

    /**
     * @brief Network constructor with pre-defined UID.
     * 
     * @param uid network UID.
     */
    explicit Network(const core::UID &uid) : base_({uid, {}}) {}

public:
    /**
     * @brief Add a population to the network.
     * 
     * @param population population to add.
     */
    void add_population(core::AllPopulationsVariant &&population);
    /**
     * @brief Add a population to the network.
     * 
     * @tparam PopulationType type of population to add (derived automatically from @p population if not specified).
     * 
     * @param population population to add.
     */
    template <typename PopulationType>
    void add_population(typename std::decay<PopulationType>::type &&population);
    /**
     * @brief Add a population to the network.
     * 
     * @tparam PopulationType type of population to add (derived automatically from @p population if not specified).
     * 
     * @param population population to add.
     */
    template <typename PopulationType>
    void add_population(typename std::decay<PopulationType>::type &population);

    /**
     * @brief Check if population exists.
     * 
     * @param population_uid UID of the population to check.
     * 
     * @return `true` if population exists, `false` otherwise.
     */
    [[nodiscard]] bool is_population_exists(const knp::core::UID &population_uid) const;

    /**
     * @brief Get a population with the given UID from the network.
     * 
     * @tparam NeuronType type of population neuron.
     * 
     * @param population_uid population UID.
     * 
     * @return population.
     * 
     * @throw std::logic_error if population is not found in the network.
     */
    template <typename NeuronType>
    [[nodiscard]] knp::core::Population<NeuronType> &get_population(const knp::core::UID &population_uid);

    /**
     * @brief Get a population with the given UID from the network.
     * 
     * @tparam NeuronType type of population neuron.
     * 
     * @param population_uid population UID.
     * 
     * @return population.
     * 
     * @note Constant method.
     * 
     * @throw std::logic_error if population is not found in the network.
     */
    template <typename NeuronType>
    [[nodiscard]] const knp::core::Population<NeuronType> &get_population(const knp::core::UID &population_uid) const;

    /**
     * @brief Get a population with the given UID from the network.
     * 
     * @param population_uid population UID.
     * 
     * @return population.
     * 
     * @throw std::logic_error if population is not found in the network.
     */
    [[nodiscard]] core::AllPopulationsVariant &get_population(const knp::core::UID &population_uid);

    /**
     * @brief Remove a population with the given UID from the network.
     * 
     * @param population_uid UID of the population to remove.
     */
    void remove_population(const knp::core::UID &population_uid);

public:
    /**
     * @brief Add a projection to the network.
     * 
     * @param projection projection to add.
     */
    void add_projection(core::AllProjectionsVariant &&projection);
    /**
     * @brief Add a projection to the network.
     * 
     * @tparam ProjectionType type of projection to add (derived automatically from @p projection if not specified).
     * 
     * @param projection projection to add.
     */
    template <typename ProjectionType>
    void add_projection(typename std::decay<ProjectionType>::type &&projection);
    /**
     * @brief Add a projection to the network.
     * 
     * @tparam ProjectionType type of projection to add (derived automatically from @p projection if not specified).
     * 
     * @param projection projection to add.
     */
    template <typename ProjectionType>
    void add_projection(typename std::decay<ProjectionType>::type &projection);
    /**
     * @brief Add a projection to the network.
     * 
     * @tparam SynapseType type of projection synapses.
     * 
     * @param projection_uid UID of projection to add.
     * @param generator synapse generator.
     * @param pre_population_uid UID of the presynaptic population.
     * @param post_population_uid UID of the postsynaptic population.
     * @param synapse_count number of projection synapses.
     */
    template <typename SynapseType>
    void add_projection(
        knp::core::UID projection_uid, knp::core::UID pre_population_uid, knp::core::UID post_population_uid,
        typename knp::core::Projection<SynapseType>::SynapseGenerator generator, size_t synapse_count);

    /**
     * @brief Check if projection exists.
     * 
     * @param projection_uid UID of the projection to check.
     * 
     * @return `true` if projection exists, `false` otherwise.
     */
    [[nodiscard]] bool is_projection_exists(const knp::core::UID &projection_uid) const;

    /**
     * @brief Get a projection with the given UID from the network.
     * 
     * @tparam SynapseType type of projection synapse.
     * 
     * @param projection_uid projection UID.
     * 
     * @return projection.
     * 
     * @throw std::logic_error if projection is not found in the network.
     */
    template <typename SynapseType>
    [[nodiscard]] knp::core::Projection<SynapseType> &get_projection(const knp::core::UID &projection_uid);
    /**
     * @brief Get a projection with the given UID from the network.
     * 
     * @tparam SynapseType type of projection synapse.
     * 
     * @param projection_uid projection UID.
     * 
     * @return projection.
     * 
     * @note Constant method.
     * 
     * @throw std::logic_error if projection is not found in the network.
     */
    template <typename SynapseType>
    [[nodiscard]] const knp::core::Projection<SynapseType> &get_projection(const knp::core::UID &projection_uid) const;
    /**
     * @brief Get a projection with the given UID from the network.
     * 
     * @param projection_uid projection UID.
     * 
     * @return projection.
     * 
     * @throw std::logic_error if projection is not found in the network.
     */
    [[nodiscard]] core::AllProjectionsVariant &get_projection(const knp::core::UID &projection_uid);
    /**
     * @brief Remove a projection with the given UID from the network.
     * 
     * @param projection_uid UID of the projection to remove.
     */
    void remove_projection(const knp::core::UID &projection_uid);

public:
    /**
     * @brief Connect presynaptic and postsynaptic populations and add projection to the network.
     * 
     * @tparam SourceNeuronType type of the presynaptic population neuron.
     * @tparam DestinationNeuronType type of the postsynaptic population neuron.
     * @tparam SynapseType new projection synapse type.
     * 
     * @param src presynaptic population.
     * @param dst postsynaptic population.
     * @param syn_gen synapse parameters generator.
     * 
     * @return new projection UID.
     * 
     * @details The method makes connections between each presynaptic population (source) neuron
     * to each postsynaptic population (destination) neuron.
     */
    template <typename SynapseType, typename SourceNeuronType, typename DestinationNeuronType>
    knp::core::UID connect_populations(
        const core::Population<SourceNeuronType> &src, const core::Population<DestinationNeuronType> &dst,
        typename projection::parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
            projection::parameters_generators::default_synapse_gen_2param<SynapseType>)
    {
        const auto &[src_uid, dst_uid] = get_populations_uid(src, dst);
        auto new_proj =
            projection::connect_populations<SynapseType, SourceNeuronType, DestinationNeuronType>(src, dst, syn_gen);
        const auto proj_uid = new_proj.get_uid();

        add_projection(std::move(new_proj));

        return proj_uid;
    }

    /**
     * @brief Connect presynaptic and postsynaptic populations and add projection to the network.
     * 
     * @tparam SourceNeuronType type of the presynaptic population neuron.
     * @tparam DestinationNeuronType type of the postsynaptic population neuron.
     * @tparam SynapseType new projection synapse type.
     * 
     * @param src presynaptic population.
     * @param dst postsynaptic population.
     * @param syn_gen synapse generator.
     * @param num_iterations number of iterations.
     * 
     * @return new projection UID.
     */
    template <typename SynapseType, typename SourceNeuronType, typename DestinationNeuronType>
    knp::core::UID connect_populations(
        const core::Population<SourceNeuronType> &src, const core::Population<DestinationNeuronType> &dst,
        typename knp::core::Projection<SynapseType>::SynapseGenerator syn_gen, size_t num_iterations)
    {
        const auto &[src_uid, dst_uid] = get_populations_uid(src, dst);
        auto new_proj = projection::connect_populations<SynapseType, SourceNeuronType, DestinationNeuronType>(
            src, dst, syn_gen, num_iterations);
        const auto proj_uid = new_proj.get_uid();

        add_projection(std::move(new_proj));

        return proj_uid;
    }

public:
    /**
     * @brief Get an iterator pointing to the first element of the population.
     * 
     * @return population iterator.
     */
    [[nodiscard]] PopulationIterator begin_populations();
    /**
     * @brief Get an iterator pointing to the first element of the population.
     * 
     * @return constant population iterator.
     */
    [[nodiscard]] PopulationConstIterator begin_populations() const;
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * 
     * @return iterator.
     */
    [[nodiscard]] PopulationIterator end_populations();
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * 
     * @return constant iterator.
     */
    [[nodiscard]] PopulationConstIterator end_populations() const;
    /**
     * @brief Get an iterator pointing to the first element of the projection.
     * 
     * @return projection iterator.
     */
    [[nodiscard]] ProjectionIterator begin_projections();
    /**
     * @brief Get an iterator pointing to the first element of the projection.
     * 
     * @return constant projection iterator.
     */
    [[nodiscard]] ProjectionConstIterator begin_projections() const;
    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * 
     * @return iterator.
     */
    [[nodiscard]] ProjectionIterator end_projections();
    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * 
     * @return constant iterator.
     */
    [[nodiscard]] ProjectionConstIterator end_projections() const;

public:
    /**
     * @brief Get population container from the network.
     * 
     * @return container of populations.
     */
    [[nodiscard]] const PopulationContainer &get_populations() const { return populations_; }
    /**
     * @brief Get projection container from the network.
     * 
     * @return container of projections.
     */
    [[nodiscard]] const ProjectionContainer &get_projections() const { return projections_; }

public:
    /**
     * @brief Count populations in the network.
     * 
     * @return number of populations.
     */
    [[nodiscard]] size_t populations_count() const { return populations_.size(); }
    /**
     * @brief Count projections in the network.
     * 
     * @return number of projections.
     */
    [[nodiscard]] size_t projections_count() const { return projections_.size(); }

public:
    /**
     * @brief Get network UID.
     * 
     * @return UID.
     */
    [[nodiscard]] const knp::core::UID &get_uid() const { return base_.uid_; }

    /**
     * @brief Get tags used by the network.
     * 
     * @return tag map.
     * 
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

private:
    template <typename SourceNeuronType, typename DestinationNeuronType>
    [[nodiscard]] std::tuple<core::UID, core::UID> get_populations_uid(
        const core::Population<SourceNeuronType> &src, const core::Population<DestinationNeuronType> &dst) const
    {
        const auto src_uid = src.get_uid();
        if (!is_population_exists(src_uid))
        {
            throw std::logic_error("Source population with UID = " + std::string(src_uid) + " doesn't exist.");
        }

        const auto dst_uid = dst.get_uid();
        if (!is_population_exists(dst_uid))
        {
            throw std::logic_error("Source population with UID = " + std::string(dst_uid) + " doesn't exist.");
        }

        return std::make_tuple(src_uid, dst_uid);
    }

    template <typename PopulationType>
    void check_population_constraints(const PopulationType &population) const;
    template <typename ProjectionType>
    void check_projection_constraints(const ProjectionType &projection) const;

private:
    knp::core::BaseData base_;
    PopulationContainer populations_;
    ProjectionContainer projections_;
};


}  // namespace knp::framework
