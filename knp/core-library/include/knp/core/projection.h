/**
 * @file projection.h
 * @brief General projection interface.
 * @kaspersky_support Artiom N.
 * @date 18.01.2023
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

#include <knp/core/core.h>
#include <knp/core/uid.h>
#include <knp/synapse-traits/all_traits.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index_container.hpp>


/**
 * @brief Core library namespace.
 */
namespace knp::core
{
/**
 * @brief Enumeration used to access connection.
 */
enum SynapseElementAccess
{
    /**
     * @brief Getting synapse parameters.
     */
    synapse_data = 0,
    /**
     * @brief Getting source neuron index.
     */
    source_neuron_id = 1,
    /**
     * @brief Getting target neuron index.
     */
    target_neuron_id = 2
};


/**
 * @brief The Projection class is a definition of similar connections between the neurons of two populations.
 * @todo This class should later be divided to interface and implementation classes.
 * @tparam SynapseType type of synapses the projection contains.
 * @see ALL_SYNAPSES.
 */
template <class SynapseType>
class Projection final
{
public:
    /**
     * @brief Type of the projection synapses.
     */
    using ProjectionSynapseType = SynapseType;
    /**
     * @brief Projection of synapses with the specified synapse type.
     */
    using ProjectionType = Projection<SynapseType>;
    /**
     * @brief Parameters of the specified synapse type.
     */
    using SynapseParameters = typename synapse_traits::synapse_parameters<SynapseType>;

    /**
     * @brief Synapse description structure that contains synapse parameters and indexes of the associated neurons.
     */
    using Synapse = std::tuple<SynapseParameters, size_t, size_t>;

    /**
     * @brief Synapse generation function type.
     */
    using SynapseGenerator = std::function<std::optional<Synapse>(size_t)>;

    /**
     * @brief Type of the container that contains synapses.
     */
    using SynapsesContainer = std::vector<Synapse>;

    /**
     * @brief Iterator for synapses.
     */
    using iterator = typename SynapsesContainer::iterator;

public:
    /**
     * @brief Shared synapse parameters for the non-STDP variant of the projection.
     * @tparam SynapseT projection synapse type.
     */
    template <typename SynapseT>
    struct SharedSynapseParametersT
    {
        /**
         * @brief Shared synapse parameters.
         */
        knp::synapse_traits::shared_synapse_parameters<SynapseT> synapses_parameters_;
    };

    /**
     * @brief Structure for the parameters shared between synapses for STDP.
     * @tparam Rule STDP rule type.
     * @tparam SynapseT synapse type.
     */
    template <template <typename> typename Rule, typename SynapseT>
    struct SharedSynapseParametersT<knp::synapse_traits::STDP<Rule, SynapseT>>
    {
        /**
         * @brief STDP population messages processing type.
         */
        enum class ProcessingType
        {
            /**
             * @brief STDP messages only.
             */
            STDPOnly,
            /**
             * @brief STDP messages and spikes.
             */
            STDPAndSpike
        };

        /**
         * @brief Map type for saving STDP population UIDs and linked processing type.
         */
        using ContainerType = std::unordered_map<core::UID, ProcessingType, core::uid_hash>;
        /**
         * @brief Map to save population UID and processing type.
         */
        ContainerType stdp_populations_;

        /**
         * @brief Shared synapse parameters for the current projection.
         */
        knp::synapse_traits::shared_synapse_parameters<knp::synapse_traits::STDP<Rule, SynapseT>> synapses_parameters_;
    };

    /**
     * @brief Type of shared synapse parameters.
     */
    using SharedSynapseParameters = SharedSynapseParametersT<SynapseType>;

public:
    /**
     * @brief Construct an empty projection.
     * @param presynaptic_uid UID of the presynaptic population.
     * @param postsynaptic_uid UID of the postsynaptic population.
     */
    Projection(UID presynaptic_uid, UID postsynaptic_uid);
    /**
     * @brief Construct an empty projection.
     * @param uid projection UID.
     * @param presynaptic_uid UID of the presynaptic population.
     * @param postsynaptic_uid UID of the postsynaptic population.
     */
    Projection(UID uid, UID presynaptic_uid, UID postsynaptic_uid);

    /**
     * @brief Construct a projection by running a synapse generator a given number of times.
     * @param presynaptic_uid presynaptic population UID.
     * @param postsynaptic_uid postsynaptic population UID.
     * @param generator function that generates synapse parameters: `params_`, `id_from_`, `id_to_`.
     * @param num_iterations number of times to run the synapse generator.
     */
    Projection(UID presynaptic_uid, UID postsynaptic_uid, SynapseGenerator generator, size_t num_iterations);

    /**
     * @brief Construct a projection by running a synapse generator a given number of times.
     * @param uid projection UID.
     * @param presynaptic_uid presynaptic population UID.
     * @param postsynaptic_uid postsynaptic population UID.
     * @param generator function that generates synapse parameters: `params_`, `id_from_`, `id_to_`.
     * @param num_iterations number of times to run the synapse generator.
     */
    Projection(UID uid, UID presynaptic_uid, UID postsynaptic_uid, SynapseGenerator generator, size_t num_iterations);

public:
    /**
     * @brief Get projection UID.
     * @return projection UID.
     */
    [[nodiscard]] const UID &get_uid() const { return base_.uid_; }

    /**
     * @brief Get tags used by the projection.
     * @return projection tag map.
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

    /**
     * @brief Get tags used by the projection.
     * @return projection tag map.
     * @see TagMap.
     */
    [[nodiscard]] const auto &get_tags() const { return base_.tags_; }

public:
    /**
     * @brief Get parameter values of a synapse with the given index.
     * @param index synapse index.
     * @return synapse parameters and indexes.
     */
    [[nodiscard]] Synapse &operator[](size_t index) { return parameters_[index]; }

    /**
     * @brief Get parameter values of a synapse with the given index.
     * @details Constant method.
     * @param index synapse index.
     * @return synapse parameters and indexes.
     */
    [[nodiscard]] const Synapse &operator[](size_t index) const { return parameters_[index]; }

    /**
     * @brief Get an iterator pointing to the first element of the projection.
     * @return constant projection iterator.
     */
    [[nodiscard]] auto begin() const { return parameters_.cbegin(); }

    /**
     * @brief Get an iterator pointing to the first element of the projection.
     * @return projection iterator.
     */
    [[nodiscard]] auto begin() { return parameters_.begin(); }

    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * @return constant iterator.
     */
    [[nodiscard]] auto end() const { return parameters_.cend(); }

    /**
     * @brief Get an iterator pointing to the last element of the projection.
     * @return iterator.
     */
    [[nodiscard]] auto end() { return parameters_.end(); }

public:
    /**
     * @brief Count number of synapses in the projection.
     * @return number of synapses.
     */
    [[nodiscard]] size_t size() const { return parameters_.size(); }

    /**
     * @brief Get UID of the associated population from which this projection receives spikes.
     * @return UID of the presynaptic population.
     */
    [[nodiscard]] const UID &get_presynaptic() const { return presynaptic_uid_; }

    /**
     * @brief Get UID of the associated population to which this projection sends signals.
     * @return UID of the postsynaptic population.
     */
    [[nodiscard]] const UID &get_postsynaptic() const { return postsynaptic_uid_; }

    /**
     * @brief Types of synapse search.
     */
    enum class Search
    {
        /**
         * @brief Search by presynaptic neuron index.
         */
        by_presynaptic,
        /**
         * @brief Search by postsynaptic neuron index.
         */
        by_postsynaptic
    };

    /**
     * @brief Find synapses that originate from a neuron with the given index.
     * @param neuron_index index of a neuron.
     * @param search_method search by presynaptic or postsynaptic neuron.
     * @return indexes of all synapses associated with the specified presynaptic neuron.
     */
    [[nodiscard]] std::vector<size_t> find_synapses(size_t neuron_index, Search search_method) const;

    /**
     * @brief Append connections to the existing projection.
     * @param generator synapse generation function.
     * @param num_iterations number of iterations to run the synapse generator.
     * @return number of synapses added to the projection, which can be less or equal to the `num_iterations` value.
     */
    size_t add_synapses(SynapseGenerator generator, size_t num_iterations);

    /**
     * @brief Remove all synapses from the projection.
     */
    void clear();

    /**
     * @brief Remove a synapse with the given index from the projection.
     * @param index index of the synapse to remove.
     */
    void remove_synapse(size_t index);

    /**
     * @brief Remove synapses according to a given criterion.
     * @param predicate functor that receives a synapse and returns `true` if the synapse must be deleted.
     * @return number of deleted synapses.
     */
    size_t remove_synapse_if(std::function<bool(const Synapse &)> predicate);

    /**
     * @brief Remove all synapses that lead to a neuron with the given index.
     * @param neuron_index index of the postsynaptic neuron which related synapses must be deleted.
     * @return number of deleted synapses.
     */
    size_t remove_postsynaptic_neuron_synapses(size_t neuron_index);

    /**
     * @brief Remove all synapses that receive signals from a neuron with the given index.
     * @param neuron_index index of the presynaptic neuron which related synapses must be deleted.
     * @return number of deleted synapses.
     */
    size_t remove_presynaptic_neuron_synapses(size_t neuron_index);

public:
    /**
     * @brief Lock the possibility to change synapses weights.
     */
    void lock_weights() { is_locked_ = true; }

    /**
     * @brief Unlock the possibility to change synapses weights.
     */
    void unlock_weights() { is_locked_ = false; }

    /**
     * @brief Determine if the synapse weight change is locked.
     * @return `true` if the synapse weight change is locked, `false` if the synapse weight change is not locked.
     */
    bool is_locked() const { return is_locked_; }

public:
    /**
     * @brief Get parameters shared between all synapses.
     * @return shared parameters.
     */
    SharedSynapseParameters &get_shared_parameters() { return shared_parameters_; }
    /**
     * @brief Get parameters shared between all synapses.
     * @note Constant method.
     * @return shared parameters.
     */
    const SharedSynapseParameters &get_shared_parameters() const { return shared_parameters_; }

private:
    void reindex() const;

    BaseData base_;

    /**
     * @brief UID of the population that sends spikes to the projection (presynaptic population)
     */
    UID presynaptic_uid_;

    /**
     * @brief UID of the population that receives synapse responses from this projection (postsynaptic population).
     */
    UID postsynaptic_uid_;

    /**
     * @brief Return `false` if the weight change for synapses is not locked.
     */
    bool is_locked_ = true;

    /**
     * @brief Container of synapse parameters.
     */
    SynapsesContainer parameters_;
    // So far the index is mutable so we can reindex a const object that has a non-updated index.
    struct Connection
    {
        size_t from_;
        size_t to_;
        size_t index_;
        bool operator==(const Connection &connection) const
        {
            return from_ == connection.from_ && to_ == connection.to_ && index_ == connection.index_;
        }
    };

    typedef boost::multi_index::multi_index_container<
        Connection,
        boost::multi_index::indexed_by<
            boost::multi_index::hashed_non_unique<
                boost::multi_index::tag<struct mi_presynaptic>, BOOST_MULTI_INDEX_MEMBER(Connection, size_t, from_)>,
            boost::multi_index::hashed_non_unique<
                boost::multi_index::tag<struct mi_postsynaptic>, BOOST_MULTI_INDEX_MEMBER(Connection, size_t, to_)>,
            boost::multi_index::hashed_unique<
                boost::multi_index::tag<struct mi_synapse_index>,
                BOOST_MULTI_INDEX_MEMBER(Connection, size_t, index_)>>>
        Index;
    using ByPresynaptic = mi_presynaptic;
    using ByPostsynaptic = mi_postsynaptic;

    mutable Index index_;
    mutable bool is_index_updated_ = false;

    SharedSynapseParameters shared_parameters_;
};


/**
 * @brief List of projection types based on synapse types specified in `knp::synapse_traits::AllSynapses`.
 * @details `AllProjections` takes the value of `Projection<SynapseType_1>, Projection<SynapseType_2>, ...,
 * Projection<SynapseType_n>`, where `SynapseType_[1..n]` is the synapse type specified in
 * `knp::synapse_traits::AllSynapses`. \n For example, if `knp::synapse_traits::AllSynapses` contains DeltaSynapse and
 * AdditiveSTDPSynapse types, then `AllProjections` = `Population<DeltaSynapse>, Population<AdditiveSTDPSynapse>`.
 */
using AllProjections = boost::mp11::mp_transform<knp::core::Projection, knp::synapse_traits::AllSynapses>;

/**
 * @brief Projection variant that contains any projection type specified in `AllProjections`.
 * @details `AllProjectionVariants` takes the value of `std::variant<ProjectionType_1,..., ProjectionType_n>`, where
 * `ProjectionType_[1..n]` is the projection type specified in `AllProjections`. \n For example, if `AllProjections`
 * contains DeltaSynapse and AdditiveSTDPSynapse types, then `AllProjectionVariants = std::variant<DeltaSynapse,
 * AdditiveSTDPSynapse>`. \n `AllProjectionVariants` retains the same order of message types as defined in
 * `AllProjections`.
 * @see ALL_SYNAPSES.
 */
using AllProjectionsVariant = boost::mp11::mp_rename<AllProjections, std::variant>;


}  // namespace knp::core
