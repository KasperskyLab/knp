/**
 * @file synapse_generators.h
 * @brief Projection connectors.
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

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <optional>
#include <random>
#include <tuple>

#include "synapse_parameters_generators.h"


/**
 * @brief Synapse generators namespace.
 * @detail Generators make connections between presynaptic population neurons to postsynaptic population neurons.
 */
namespace knp::framework::projection::synapse_generators
{

/**
 * @brief For populations of size `N x M` generates connections such as: `0 ->
 * 0`, `0 -> 1`, `0 -> 2`, ..., `0 -> M`, `1 -> 0`, `1 -> 1`, ..., `1 -> M`, ..., `N -> M`.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return synapse generator.
 */
template <typename SynapseType>
[[nodiscard]] typename knp::core::Projection<SynapseType>::SynapseGenerator all_to_all(
    size_t presynaptic_pop_size, size_t postsynaptic_pop_size,
    const parameters_generators::SynGen2ParamsType<SynapseType> &syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    return [presynaptic_pop_size, postsynaptic_pop_size,
            syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
    {
        const size_t index0 = index % presynaptic_pop_size;
        const size_t index1 = index / presynaptic_pop_size;

        return std::make_tuple(syn_gen(index0, index1), index0, index1);
    };
}


/**
 * @brief For example if population0_size is 2 and population1_size is 4, then synapses amount
 * must be 4, and generator will create synapses as follows: 0-0, 0-1, 1-2, 1-3. So generator will distribute
 * connections evenly.
 * @param presynaptic_pop_size size of first population.
 * @param postsynaptic_pop_size size of second population.
 * @param syn_gen generator of synapse parameters.
 * @return synapse generator.
 * tparam SynapseType projection synapse type.
 */
template <typename SynapseType>
[[nodiscard]] typename knp::core::Projection<SynapseType>::SynapseGenerator aligned(
    size_t presynaptic_pop_size, size_t postsynaptic_pop_size,
    parameters_generators::SynGen2ParamsType<SynapseType> const &syn_gen =
        parameters_generators::default_synapse_gen<SynapseType>)
{
    return [presynaptic_pop_size, postsynaptic_pop_size,
            syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
    {
        size_t from_index;
        size_t pack_size;
        size_t to_index;

        if (presynaptic_pop_size >= postsynaptic_pop_size)
        {
            from_index = index;
            pack_size = presynaptic_pop_size / postsynaptic_pop_size;
            to_index = index / pack_size;
        }
        else
        {
            to_index = index;
            pack_size = postsynaptic_pop_size / presynaptic_pop_size;
            from_index = index / pack_size;
        }
        return std::make_tuple(syn_gen(from_index, to_index), from_index, to_index);
    };
}


/**
 * @brief For example if populations size is 3, then synapses amount is 6,
 * and generator will generate synapses as follows: 0-1, 0-2, 1-0, 1-2, 2-0, 2-1. So it excludes one synapse at a time.
 * @pre Population sizes must be equal.
 * @param populations_size size of populations, they are supposed to be the same.
 * @param syn_gen generator of synapse parameters.
 * @return synapse generator.
 * tparam SynapseType projection synapse type.
 */
template <typename SynapseType>
[[nodiscard]] typename knp::core::Projection<SynapseType>::SynapseGenerator exclusive(
    size_t populations_size, parameters_generators::SynGen2ParamsType<SynapseType> const &syn_gen =
                                 parameters_generators::default_synapse_gen<SynapseType>)
{
    return
        [populations_size, syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
    {
        size_t from_index;
        size_t to_index;
        from_index = index / (populations_size - 1);
        to_index = index % (populations_size - 1);
        if (to_index >= from_index) ++to_index;
        return std::make_tuple(syn_gen(from_index, to_index), from_index, to_index);
    };
}


/**
 * @brief For the populations of size `N x N` generates connections such as: `0 -> 0`,
 * `1 -> 1`, `2 -> 2`, ..., `N -> N`.
 * @pre Population sizes must be equal.
 * @param population_size neuron count in populations.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return synapse generator.
 */
template <typename SynapseType>
[[nodiscard]] typename knp::core::Projection<SynapseType>::SynapseGenerator one_to_one(
    size_t population_size, parameters_generators::SynGen1ParamType<SynapseType> syn_gen = std::bind(
                                parameters_generators::default_synapse_gen<SynapseType>, std::placeholders::_1, 0))
{
    return [syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
    { return std::make_tuple(syn_gen(index), index, index); };
}


/**
 * @brief Generate synapses from container.
 * @details Container must contain synapses as `(parameters, from_index, to_index)` tuples,
 * where `parameters` are synapse parameters, `from_index` is presynaptic neuron index,
 * and `to_index` is postsynaptic neuron index.
 * @param container container with synapses.
 * @tparam SynapseType neuron type.
 * @tparam Container container type.
 * @return synapse generator.
 */
template <typename SynapseType, template <typename...> class Container>
[[nodiscard]] typename knp::core::Projection<SynapseType>::SynapseGenerator from_container(
    const Container<typename core::Projection<SynapseType>::Synapse> &container)
{
    return [&container](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
    { return container[index]; };
}


/**
 * @brief The FromMap class is a defintion of a generator of synapses from `std::map` or `std::unordered_map` object.
 * @details 'std::map' object must contain synapse parameters as values and `(from_index, to_index)` tuples as keys.
 * @tparam SynapseType projection synapse type.
 * @tparam Map map class.
 */
template <typename SynapseType, template <typename, typename, typename...> class Map>
class FromMap
{
public:
    /**
     * @brief Type of the map.
     */
    using MapType =
        Map<typename std::tuple<size_t, size_t>, typename knp::core::Projection<SynapseType>::SynapseParameters>;

    /**
     * @brief Constructor.
     * @param synapses_map map with tuples containing indexes of presynaptic and postsynaptic neurons as keys
     *  and synapse parameters as values.
     */
    explicit FromMap(const MapType &synapses_map) : iter_(synapses_map.cbegin()) {}

    /**
     * @brief Call operator.
     * @param index synapse index.
     * @return optional synapse parameters.
     */
    [[nodiscard]] typename std::optional<typename knp::core::Projection<SynapseType>::Synapse> operator()(size_t index)
    {
        const auto [from_index, to_index] = iter_->first;
        auto synapse = std::make_tuple(iter_->second, from_index, to_index);
        ++iter_;
        return synapse;
    }

private:
    typename MapType::const_iterator iter_;
};


/**
 * @brief The FixedProbability class is a definition of a generator that makes connections with some probability
 * between each presynaptic population (source) neuron to each postsynaptic population (destination) neuron.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 */
template <typename SynapseType>
class FixedProbability
{
public:
    /**
     * @brief Constructor.
     * @param presynaptic_pop_size presynaptic population neuron count.
     * @param postsynaptic_pop_size postsynaptic population neuron count.
     * @param connection_probability connection probability.
     * @param syn_gen generator of synapse parameters.
     */
    FixedProbability(
        size_t presynaptic_pop_size, size_t postsynaptic_pop_size, double connection_probability,
        parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
            parameters_generators::default_synapse_gen<SynapseType>)
        : presynaptic_pop_size_(presynaptic_pop_size),
          postsynaptic_pop_size_(postsynaptic_pop_size),
          connection_probability_(connection_probability),
          syn_gen_(syn_gen),
          mt_(std::random_device()()),
          dist_(0, 1)
    {
        if (connection_probability > 1 || connection_probability < 0)
            throw std::logic_error("Incorrect probability, set probability between 0 and 1.");
    }

    /**
     * @brief Call operator.
     * @param index synapse index.
     * @return optional synapse parameters.
     */
    [[nodiscard]] typename std::optional<typename knp::core::Projection<SynapseType>::Synapse> operator()(size_t index)
    {
        const size_t index0 = index % presynaptic_pop_size_;
        const size_t index1 = index / presynaptic_pop_size_;

        if (dist_(mt_) < connection_probability_) return std::make_tuple(syn_gen_(index0, index1), index0, index1);
        return std::nullopt;
    }

private:
    size_t presynaptic_pop_size_;
    size_t postsynaptic_pop_size_;
    double connection_probability_;
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen_;
    std::mt19937 mt_;
    std::uniform_real_distribution<double> dist_;
};


/**
 * @brief Make connections between neurons of presynaptic and postsynaptic populations
 * based on the synapse generation function result.
 * @param presynaptic_pop_size presynaptic population neuron count.
 * @param postsynaptic_pop_size postsynaptic population neuron count.
 * @param syn_gen generator of synapse parameters.
 * @tparam SynapseType projection synapse type.
 * @return synapse generator.
 */
template <typename SynapseType>
[[nodiscard]] typename knp::core::Projection<SynapseType>::SynapseGenerator index_based(
    size_t presynaptic_pop_size, size_t postsynaptic_pop_size,
    parameters_generators::SynGenOptional2ParamsType<SynapseType> syn_gen)
{
    return [presynaptic_pop_size, postsynaptic_pop_size,
            syn_gen](size_t index) -> std::optional<typename knp::core::Projection<SynapseType>::Synapse>
    {
        const size_t index0 = index % presynaptic_pop_size;
        const size_t index1 = index / presynaptic_pop_size;
        auto opt_result = syn_gen(index0, index1);

        if (opt_result.has_value()) return std::make_tuple(opt_result.value(), index0, index1);
        return std::nullopt;
    };
}


/**
 * @brief The FixedNumberPost class is a definition of a generator that makes connections between each presynaptic
 * neuron and a fixed number of random postsynaptic neurons.
 * @details This connector uses MT19937 generator with uniform integer distribution.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 */
template <typename SynapseType>
class FixedNumberPost
{
public:
    /**
     * @brief Constructor.
     * @param presynaptic_pop_size presynaptic population neuron count.
     * @param postsynaptic_pop_size postsynaptic population neuron count.
     * @param syn_gen generator of synapse parameters.
     */
    FixedNumberPost(
        size_t presynaptic_pop_size, size_t postsynaptic_pop_size,
        std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t index0, size_t index1)>
            syn_gen = parameters_generators::default_synapse_gen<SynapseType>)
        : presynaptic_pop_size_(presynaptic_pop_size),
          postsynaptic_pop_size_(postsynaptic_pop_size),
          syn_gen_(syn_gen),
          mt_(std::random_device()()),
          dist_(0, postsynaptic_pop_size - 1)
    {
    }

    /**
     * @brief Call operator.
     * @param index synapse index.
     * @return optional synapse parameters.
     */
    [[nodiscard]] typename std::optional<typename knp::core::Projection<SynapseType>::Synapse> operator()(size_t index)
    {
        const size_t index0 = index % presynaptic_pop_size_;
        const size_t index1 = dist_(mt_);

        return std::make_tuple(syn_gen_(index0, index1), index0, index1);
    }

private:
    size_t presynaptic_pop_size_;
    size_t postsynaptic_pop_size_;
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen_;
    std::mt19937 mt_;
    std::uniform_int_distribution<size_t> dist_;
};


/**
 * @brief The FixedNumberPre class is a definition of a generator that makes connections between each postsynaptic
 * neuron and a fixed number of random presynaptic neurons.
 * @details This uses MT19937 generator with uniform integer distribution.
 * @warning It doesn't get "real" populations and can't be used with populations that contain non-contiguous indexes.
 * @tparam SynapseType projection synapse type.
 */
template <typename SynapseType>
class FixedNumberPre
{
public:
    /**
     * @brief Constructor.
     * @param presynaptic_pop_size presynaptic population neuron count.
     * @param postsynaptic_pop_size postsynaptic population neuron count.
     * @param syn_gen generator of synapse parameters.
     */
    FixedNumberPre(
        size_t presynaptic_pop_size, size_t postsynaptic_pop_size,
        std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t index0, size_t index1)>
            syn_gen = parameters_generators::default_synapse_gen<SynapseType>)
        : presynaptic_pop_size_(presynaptic_pop_size),
          postsynaptic_pop_size_(postsynaptic_pop_size),
          syn_gen_(syn_gen),
          mt_(std::random_device()()),
          dist_(0, presynaptic_pop_size - 1)
    {
    }

    /**
     * @brief Call operator.
     * @param index synapse index.
     * @return optional synapse parameters.
     */
    [[nodiscard]] typename std::optional<typename knp::core::Projection<SynapseType>::Synapse> operator()(size_t index)
    {
        const size_t index0 = dist_(mt_);
        const size_t index1 = index % postsynaptic_pop_size_;

        return std::make_tuple(syn_gen_(index0, index1), index0, index1);
    }

private:
    size_t presynaptic_pop_size_;
    size_t postsynaptic_pop_size_;
    parameters_generators::SynGen2ParamsType<SynapseType> syn_gen_;
    std::mt19937 mt_;
    std::uniform_int_distribution<size_t> dist_;
};


/**
 * @brief Make connections duplicated from another projection.
 * @details Source and target projections can have different types.
 * In this case projection types cannot be same.
 * @todo Clone synapse parameters when projection types are the same.
 * @param source_proj source projection.
 * @param syn_gen generator of synapse parameters.
 * @tparam DestinationSynapseType generator of target synapse parameters.
 * @tparam SourceSynapseType source projection synapse type.
 * @return synapse generator.
 */
template <typename DestinationSynapseType, typename SourceSynapseType>
[[nodiscard]] typename knp::core::Projection<DestinationSynapseType>::SynapseGenerator clone_projection(
    const knp::core::Projection<SourceSynapseType> &source_proj,
    parameters_generators::SynGen1ParamType<DestinationSynapseType> syn_gen =
        parameters_generators::default_synapse_gen<DestinationSynapseType>)
{
    return [&source_proj,
            syn_gen](size_t index) -> std::optional<typename knp::core::Projection<DestinationSynapseType>::Synapse>
    {
        auto const &synapse = source_proj[index];
        return std::make_tuple(
            syn_gen(index), std::get<knp::core::source_neuron_id>(synapse),
            std::get<knp::core::target_neuron_id>(synapse));
    };
}

}  // namespace knp::framework::projection::synapse_generators
