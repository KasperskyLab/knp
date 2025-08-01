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
 * @detail Creators make generators
 */
namespace knp::framework::projection::creators
{

/**
 * @brief For populations of size `N x M` generates connections such as: `0 ->
 * 0`, `0 -> 1`, `0 -> 2`, ..., `0 -> M`, `1 -> 0`, `1 -> 1`, ..., `1 -> M`, ..., `N -> M`.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
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
 * @brief For example if population0_size is 2 and population1_size is 4, then synapses amount
 * must be 4, and generator will create synapses as follows: 0-0, 0-1, 1-2, 1-3. So generator will distribute
 * connections evenly.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size size of first population
 * @param postsynaptic_pop_size size of second population
 * @param syn_gen generator of synapse parameters
 * @return projection
 * tparam SynapseType projection synapse type
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
 * @brief For example if populations size is 3, then synapses amount is 6,
 * and generator will generate synapses as follows: 0-1, 0-2, 1-0, 1-2, 2-0, 2-1. So it excludes one synapse at a time.
 * @pre Population sizes must be equal.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param pops_size size of populations, they are supposed to be the same
 * @param syn_gen generator of synapse parameters
 * @return projection
 * tparam SynapseType projection synapse type
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
 * @brief For the populations of size `N x N` generates connections such as: `0 -> 0`,
 * `1 -> 1`, `2 -> 2`, ..., `N -> N`.
 * @pre Population sizes must be equal.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param population_size neuron count in populations.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
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
 * @brief Generate projection from container.
 * @details Container must contain synapses as `(parameters, from_index, to_index)` tuples,
 * where `parameters` are synapse parameters, `from_index` is presynaptic neuron index,
 * and `to_index` is postsynaptic neuron index.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param container container with synapses.
 * @tparam SynapseType neuron type.
 * @tparam Container container type.
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
 * @brief Generate projection from `std::map` object.
 * @details 'std::map' object must contain synapse parameters as values and `(from_index, to_index)` tuples as keys.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param synapses_map map with tuples containing indexes of presynaptic and postsynaptic neurons as keys
 *  and synapse parameters as values.
 * @tparam SynapseType projection synapse type.
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
 * @brief Make connections with some probability between each presynaptic population (source) neuron
 * to each postsynaptic population (destination) neuron.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param connection_probability connection probability.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
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
 * @brief Make connections between neurons of presynaptic and postsynaptic populations
 * based on the synapse generation function result.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
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
 * @brief Make connections between each presynaptic neuron and a fixed number of random postsynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param neurons_count number of postsynaptic neurons.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
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
 * @brief Make connections between each postsynaptic neuron and a fixed number of random presynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @param presynaptic_uid presynaptic population UID.
 * @param postsynaptic_uid postsynaptic population UID.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param neurons_count number of presynaptic neurons.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
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
 * @brief Generate a projection which connections are duplicated from another projection.
 * @details Source and target projections can have different types.
 *          In this case synapse parameters will not be cloned.
 * @todo Clone synapse parameters when projection types are the same.
 * @param source_proj source projection.
 * @param presynaptic_uid optional presynaptic population UID.
 * @param postsynaptic_uid optional postsynaptic population UID.
 * @param syn_gen generator of synapse parameters.
 * @tparam DestinationSynapseType generator of target synapse parameters.
 * @tparam SourceSynapseType source projection synapse type.
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
