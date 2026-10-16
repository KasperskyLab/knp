/**
 * @file network_constructor.h
 * @brief Functions for network construction.
 * @kaspersky_support D. Postnikov
 * @date 05.02.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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
#include <knp/framework/tags/name.h>

#include <list>
#include <string>
#include <utility>

// cppcheck-suppress missingInclude
#include "annotated_network.h"


/**
 * @brief Enumeration for population roles in the network.
 * 
 * @details Defines the functional role of each population in the spiking neural network, which determines how 
 * populations are treated during training and inference.
 */
enum class PopulationRole
{
    /**
     * @brief Classification output neurons.
     */
    OUTPUT,
    /**
     * @brief Input data processing neurons.
     */
    INPUT,
    /**
     * @brief Regular intermediate neurons.
     */
    NORMAL,
    /**
     * @brief Specialized channelized populations for image processing.
     */
    CHANNELED,
};


/**
 * @brief Information structure for network populations.
 * 
 * @details Stores metadata about network populations including their role, lifecycle considerations, and 
 * identification information for proper network management.
 */
struct PopulationInfo
{
    /**
     * @brief Role of the population in the network architecture.
     */
    PopulationRole role_;
    /**
     * @brief Flag indicating whether this population should be retained during inference.
     */
    bool keep_in_inference_;
    /**
     * @brief Number of neurons in the population.
     */
    size_t neurons_amount_;
    /**
     * @brief Population UID.
     */
    knp::core::UID uid_;
    /**
     * @brief Population name for logging and debugging purposes.
     */
    std::string name_;
};


/**
 * @brief Helper class for constructing neural networks with proper annotation.
 * 
 * @details The `NetworkConstructor class` provides a high-level interface for building neural networks while 
 * automatically managing metadata required for inference, training, and monitoring. It handles population 
 * creation, projection establishment, and proper annotation of network components for later use in inference 
 * operations.
 * 
 * This class simplifies network construction by automatically tracking which populations and projections 
 * should be retained during inference and managing the necessary metadata for WTA (Winner-Take-All) mechanisms 
 * and other network features.
 */
class NetworkConstructor
{
public:
    /**
     * @brief Constructor.
     * 
     * @details Initializes the network constructor with a reference to the annotated network that will be built 
     * during construction.
     * 
     * @param network reference to the annotated network to be constructed.
     */
    explicit NetworkConstructor(AnnotatedNetwork &network) : network_(network) {}

    /**
     * @brief Add a population to the network and record its metadata.
     * 
     * @details Creates a new population with specified neuron parameters and adds it to the network. Automatically 
     * records population metadata for inference management and role-based processing.
     * 
     * @tparam Neuron neuron type.
     * 
     * @param neuron neuron parameters for population construction.
     * @param neurons_amount number of neurons in the population.
     * @param role role of the population in the network architecture.
     * @param keep_in_inference flag indicating whether population should persist during inference.
     * @param name population name (used for logging).
     * 
     * @return reference to the saved population information structure.
     */
    template <typename Neuron>
    [[nodiscard]] const PopulationInfo &add_population(
        const knp::neuron_traits::neuron_parameters<Neuron> &neuron, size_t neurons_amount, PopulationRole role,
        bool keep_in_inference, const std::string &name)
    {
        PopulationInfo pop_info{role, keep_in_inference, neurons_amount, {}, name};
        auto pop = knp::core::Population<Neuron>(
            pop_info.uid_, [&neuron](size_t index) { return neuron; }, pop_info.neurons_amount_);
        knp::framework::tags::set_name(pop, pop_info.name_);
        network_.network_.add_population(std::move(pop));
        if (pop_info.keep_in_inference_) network_.data_.inference_population_uids_.insert(pop_info.uid_);
        if (PopulationRole::OUTPUT == pop_info.role_) network_.data_.output_uids_.push_back(pop_info.uid_);
        return pops_.emplace_back(pop_info);
    }

    /**
     * @brief Add a channelized population for specialized data processing.
     * 
     * @details Creates a channelized population that is not actually added to the network but is tracked for automatic 
     * marking and identification purposes. Channelized populations are typically used for image data processing where 
     * each neuron represents a specific channel or feature.
     * 
     * @param neurons_amount number of neurons in the channelized population.
     * @param keep_in_inference flag indicating whether population should persist during inference.
     * 
     * @return reference to the saved population information structure.
     */
    [[nodiscard]] const PopulationInfo &add_channeled_population(size_t neurons_amount, bool keep_in_inference)
    {
        return pops_.emplace_back(
            PopulationInfo{PopulationRole::CHANNELED, keep_in_inference, neurons_amount, knp::core::UID(false), ""});
    }

    /**
     * @brief Add a projection to the network.
     * 
     * @details Creates a new synaptic projection between two populations with specified synapse parameters and 
     * connection pattern. Automatically handles training flags, WTA connectivity, and inference retention management.
     * 
     * @tparam Synapse synapse type.
     * @tparam Creator yype of callable creator of synapses.
     * 
     * @param synapse synapse parameters for projection construction.
     * @param creator callable creator function for synapses (typically from `knp::framework::projection::creators`).
     * @param pop_pre presynaptic population reference.
     * @param pop_post postsynaptic population reference.
     * @param trainable flag indicating whether the projection should be trainable.
     * @param have_wta flag indicating whether this projection connects to a WTA mechanism.
     * 
     * @return UID of the created projection for later reference.
     */
    template <typename Synapse, typename Creator>
    knp::core::UID add_projection(
        const knp::synapse_traits::synapse_parameters<Synapse> &synapse, Creator creator, const PopulationInfo &pop_pre,
        const PopulationInfo &pop_post, bool trainable, bool have_wta)
    {
        knp::core::Projection<Synapse> projection = creator(
            have_wta ? knp::core::UID(false) : pop_pre.uid_, pop_post.uid_, pop_pre.neurons_amount_,
            pop_post.neurons_amount_, [&synapse](size_t, size_t) { return synapse; });

        if (trainable) projection.unlock_weights();
        network_.network_.add_projection(projection);
        if (pop_pre.keep_in_inference_ && pop_post.keep_in_inference_)
            network_.data_.inference_internal_projection_.insert(projection.get_uid());
        return projection.get_uid();
    }

private:
    std::list<PopulationInfo> pops_;
    AnnotatedNetwork &network_;
};
