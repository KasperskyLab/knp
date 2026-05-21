/**
 * @file creators.h
 * @brief Population creators.
 * @date 08.08.2024
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

#include <cinttypes>
#include <optional>
#include <random>

#include "neuron_parameters_generators.h"


/**
 * @brief Namespace for framework population routines.
 */
namespace knp::framework::population
{

/**
 * @brief Namespace for population creators.
 */
namespace creators
{
/**
 * @brief Generate a population that gets parameter values from a container.
 * 
 * @tparam NeuronType neuron type.
 * @tparam Container container type.
 * 
 * @param container container with neuron parameter values.
 *
 * @return population.
 */
template <typename NeuronType, template <typename...> class Container>
[[nodiscard]] typename core::Population<NeuronType> from_container(
    const Container<typename core::Population<NeuronType>::NeuronParameters>& container)
{
    return core::Population<NeuronType>(
        neurons_generators::from_container<NeuronType, Container>(container), container.size());
}


/**
 * @brief Generate a population with random neuron parameter values.
 * 
 * @tparam NeuronType type of neuron parameters.
 * 
 * @param neuron_count number of neurons in a population.
 * 
 * @return population.
 * 
 * @details This generator uses MT19937 RNG with uniform integer distribution.
 * 
 * @warning Neuron parameter values are absolutely random: generator doesn't pay attention to the limits.
 */
template <typename NeuronType>
[[nodiscard]] typename core::Population<NeuronType> make_random(size_t neuron_count)
{
    return core::Population<NeuronType>(neurons_generators::MakeRandom<NeuronType>(), neuron_count);
}


/**
 * @brief Generate a population with default neuron parameter values.
 * 
 * @tparam NeuronType type of neuron parameters.
 * 
 * @param neuron_count number of neurons in a population.
 * 
 * @return population.
 */
template <typename NeuronType>
[[nodiscard]] typename core::Population<NeuronType> make_default(size_t neuron_count)
{
    return core::Population<NeuronType>(neurons_generators::make_default<NeuronType>(), neuron_count);
}


/**
 * @brief Generate a population that clones neuron parameter values from the specified neuron.
 * 
 * @tparam NeuronType type of neuron parameters.
 * 
 * @param neuron_count number of neurons in a population.
 * @param source_neuron source neuron from which to clone parameters.
 * 
 * @return population.
 */
template <typename NeuronType>
[[nodiscard]] typename core::Population<NeuronType> make_clone(
    size_t neuron_count, const typename core::Population<NeuronType>::NeuronParameters& source_neuron)
{
    return core::Population<NeuronType>(neurons_generators::make_clone<NeuronType>(source_neuron), neuron_count);
}

}  // namespace creators

}  // namespace knp::framework::population
