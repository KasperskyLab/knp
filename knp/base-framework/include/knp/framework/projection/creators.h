/**
 * @file creators.h
 * @brief Projection creators.
 * @kaspersky_support Artiom N.
 * @date 10.08.2024
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

#include <knp/core/projection.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <optional>
#include <random>
#include <tuple>

#include "synapse_generators.h"
#include "synapse_parameters_generators.h"


/**
 * @brief Namespace for framework projection creators.
 * @details Creators make generators.
 */
namespace knp::framework::projection::creators
{

/**
 * @brief Create a projection where every neuron in the presynaptic population is connected to every neuron in the 
 * postsynaptic population.
 * @details For populations of size `N x M` the method generates connections such as: `0 ->
 * 0`, `0 -> 1`, `0 -> 2`, ..., `0 -> M`, `1 -> 0`, `1 -> 1`, ..., `1 -> M`, ..., `N -> M`.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population size.
 * @param postsynaptic_pop_size postsynaptic population size.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> all_to_all(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size,
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        synapse_generators::all_to_all<SynapseType>(presynaptic_pop_size, postsynaptic_pop_size, syn_gen),
        presynaptic_pop_size * postsynaptic_pop_size);
}


/**
 * @brief Create a projection where neurons in the presynaptic population are connected to neurons in the postsynaptic 
 * population in an aligned manner.
 * @details This function generates a projection with aligned connections, meaning that neurons from a population with 
 * a less size have consequent connections with neurons from the other population, and the number of connections for 
 * each neuron of a population with less size is determined by that size.
 * For example, if the presynaptic population has 2 neurons and the postsynaptic population has 4 neurons, the connections 
 * will be: (0, 0), (0, 1), (1, 2), (1, 3).
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size size of the presynaptic population.
 * @param postsynaptic_pop_size size of the postsynaptic population.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> aligned(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size,
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        synapse_generators::aligned<SynapseType>(presynaptic_pop_size, postsynaptic_pop_size, syn_gen),
        std::max(presynaptic_pop_size, postsynaptic_pop_size));
}


/**
 * @brief Create a projection where each presynaptic population neuron to each postsynaptic population neuron with
 * exception of neurons whose indexes are the same.
 * @details For example, if the population size is 3, the connections will be: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1).
 * @pre The presynaptic and postsynaptic populations must have the same size.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param pops_size size of populations.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> exclusive(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t pops_size,
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid, synapse_generators::exclusive<SynapseType>(pops_size, syn_gen),
        pops_size * (pops_size - 1));
}


/**
 * @brief Create a projection where every neuron in the presynaptic population is connected to the neuron with the same index.
 * @details For example, if the population size is 3, the connections will be: (0, 0), (1, 1), (2, 2).
 * @pre The presynaptic and postsynaptic populations must have the same size.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param population_size size of populations.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> one_to_one(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t population_size,
    parameters_generators::SynGen1ParamType<SynapseType> syn_gen =
        std::bind(parameters_generators::default_synapse_gen<SynapseType>, std::placeholders::_1, 0))
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid, synapse_generators::one_to_one<SynapseType>(population_size, syn_gen),
        population_size);
}


/**
 * @brief Create a projection from container.
 * @details Container must contain synapses as `(parameters, from_index, to_index)` tuples,
 * where `parameters` are synapse parameters, `from_index` is presynaptic neuron index,
 * and `to_index` is postsynaptic neuron index.
 * @tparam SynapseType neuron type.
 * @tparam Container container type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param container container with synapses.
 * @return projection.
 */
template <typename SynapseType, template <typename...> class Container>
[[nodiscard]] knp::core::Projection<SynapseType> from_container(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid,
    const Container<typename core::Projection<SynapseType>::Synapse> &container)
{
    return typename knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid, synapse_generators::from_container<SynapseType, Container>(container),
        container.size());
}


/**
 * @brief Create a projection from `std::map` object.
 * @details 'std::map' object must contain synapse parameters as values and `(from_index, to_index)` tuples as keys.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param synapses_map map with tuples containing indexes of presynaptic and postsynaptic neurons as keys
 *  and synapse parameters as values.
 * @return projection.
 */
template <typename SynapseType, template <typename, typename, typename...> class Map>
[[nodiscard]] knp::core::Projection<SynapseType> from_map(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid,
    const Map<typename std::tuple<size_t, size_t>, typename knp::core::Projection<SynapseType>::SynapseParameters>
        &synapses_map)
{
    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid, synapse_generators::FromMap<SynapseType, Map>(synapses_map),
        synapses_map.size());
}


/**
 * @brief Create a projection where connections between neurons in the presynaptic and postsynaptic populations 
 * are established based on a fixed probability.
 * @details The connection probability is specified by @p connection_probability which must be a value between 0 and 1.
 * @note The actual number of connections in the projection may vary due to the random nature of the connection process.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population size.
 * @param postsynaptic_pop_size postsynaptic population size.
 * @param connection_probability probability of a connection between two neurons.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> fixed_probability(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, double connection_probability,
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    const auto proj_size = presynaptic_pop_size * postsynaptic_pop_size;
    auto fp = synapse_generators::FixedProbability<SynapseType>{
        presynaptic_pop_size, postsynaptic_pop_size, connection_probability, syn_gen};

    return knp::core::Projection<SynapseType>(presynaptic_uid, postsynaptic_uid, fp, proj_size);
}


/**
 * @brief Create a projection where connections between neurons in the presynaptic and postsynaptic populations 
 * are established based on a custom index-based connection rule.
 * @details The connection rule is specified by @p syn_gen which must be a function that takes the presynaptic 
 * and postsynaptic population sizes as input and returns a container of synapse parameters.
 * @note The actual number of connections in the projection may vary depending on the connection rule.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population size.
 * @param postsynaptic_pop_size postsynaptic population size.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> index_based(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, parameters_generators::SynGenOptional2ParamsType<SynapseType> syn_gen)
{
    const auto proj_size = presynaptic_pop_size * postsynaptic_pop_size;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        synapse_generators::index_based<SynapseType>(presynaptic_pop_size, postsynaptic_pop_size, syn_gen), proj_size);
}


/**
 * @brief Create a projection where each neuron in the presynaptic population is connected to a fixed number of postsynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @note The actual number of connections in the projection may vary depending on the number of postsynaptic neurons available.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population size.
 * @param postsynaptic_pop_size postsynaptic population size.
 * @param neurons_count number of postsynaptic neurons.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> fixed_number_post(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, size_t neurons_count,
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    const auto proj_size = presynaptic_pop_size * neurons_count;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        synapse_generators::FixedNumberPost<SynapseType>(presynaptic_pop_size, postsynaptic_pop_size, syn_gen),
        proj_size);
}


/**
 * @brief Create a projection where each neuron in the postsynaptic population is connected to a fixed number of presynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @note The actual number of connections in the projection may vary depending on the number of presynaptic neurons available.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population size.
 * @param postsynaptic_pop_size postsynaptic population size.
 * @param neurons_count number of presynaptic neurons.
 * @param syn_gen generator of synapse parameters.
 * @return projection.
 */
template <typename SynapseType>
[[nodiscard]] knp::core::Projection<SynapseType> fixed_number_pre(
    const knp::core::UID &presynaptic_uid, const knp::core::UID &postsynaptic_uid, size_t presynaptic_pop_size,
    size_t postsynaptic_pop_size, size_t neurons_count,
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    const auto proj_size = postsynaptic_pop_size * neurons_count;

    return knp::core::Projection<SynapseType>(
        presynaptic_uid, postsynaptic_uid,
        synapse_generators::FixedNumberPre<SynapseType>(presynaptic_pop_size, postsynaptic_pop_size, syn_gen),
        proj_size);
}


/**
 * @brief Create a new projection by cloning the connections from an existing projection.
 * @details Source and target projections can have different types. In this case, synapse parameters will not be cloned.
 * The new projection has the same presynaptic and postsynaptic population UIDs as the source 
 * projection, unless overridden by @p presynaptic_uid and @p postsynaptic_uid.
 * @note The actual synapse parameters in the new projection may differ from those in the source projection, depending on the generator used.
 * @todo Clone synapse parameters when projection types are the same.
 * @tparam DestinationSynapseType generator of target synapse parameters.
 * @tparam SourceSynapseType source projection synapse type.
 * @param source_proj source projection to clone.
 * @param presynaptic_uid optional presynaptic population UID.
 * @param postsynaptic_uid optional postsynaptic population UID.
 * @param syn_gen generator of synapse parameters.
 * @return projection of the `DestinationSynapseType` synapses.
 */
template <typename DestinationSynapseType, typename SourceSynapseType>
[[nodiscard]] knp::core::Projection<DestinationSynapseType> clone_projection(
    const knp::core::Projection<SourceSynapseType> &source_proj,
    parameters_generators::SynGen1ParamType<DestinationSynapseType> syn_gen =
        parameters_generators::default_synapse_gen<DestinationSynapseType>,
    const std::optional<knp::core::UID> &presynaptic_uid = std::nullopt,
    const std::optional<knp::core::UID> &postsynaptic_uid = std::nullopt)
{
    return knp::core::Projection<DestinationSynapseType>(
        presynaptic_uid.value_or(source_proj.get_presynaptic()),
        postsynaptic_uid.value_or(source_proj.get_postsynaptic()),
        synapse_generators::clone_projection<DestinationSynapseType, SourceSynapseType>(source_proj, syn_gen),
        source_proj.size());
}

}  // namespace knp::framework::projection::creators
