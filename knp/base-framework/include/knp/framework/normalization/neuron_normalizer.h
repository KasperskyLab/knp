/**
 * @file neuron_normalizer.h
 * @brief Neuron normalization functions.
 * @kaspersky_support Artiom N.
 * @date 23.07.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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
#include <knp/core/population.h>
#include <knp/framework/network.h>

#include <functional>

#include "normalizers.h"


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{

/**
 * @brief Normalize parameters of neurons in a population.
 * @tparam NeuronType type of the neuron whose parameters you want to normalize.
 * @tparam DataGetter type of the class containing method to get parameter value.
 * @tparam DataSetter type of the class containing method to set parameter value.
 * @tparam ValueCorrector type of the normalization function for neuron parameters.
 * @param population population whose neuron parameters you want to normalize.
 * @param getter class used to get neuron parameter value.
 * @param setter class used to set neuron parameter value.
 * @param corrector function used to normalize neuron parameter.
 */
template <
    typename NeuronType, template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
void normalize_neurons(
    knp::core::Population<NeuronType> &population, DataGetter<NeuronType> getter, DataSetter<NeuronType> setter,
    ValueCorrector corrector)
{
    for (auto &neuron : population)
    {
        // TODO: replace with DataGetter<ElementType>, when C++23 will be used.
        setter(neuron, corrector(getter(neuron)));
    }
}


/**
 * @brief Normalize parameters of all neurons in a network.
 * @tparam DataGetter type of the class containing method to get parameter value.
 * @tparam DataSetter type of the class containing method to set parameter value.
 * @tparam ValueCorrector type of the normalization function for neuron parameters.
 * @param network network whose population neuron parameters you want to normalize.
 * @param corrector function used to normalize neuron parameter.
 */
template <
    template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
void normalize_neurons(knp::framework::Network &network, ValueCorrector corrector)
{
    for (auto &population_variant : network.get_populations())
    {
        std::visit(
            [&corrector](auto &population)
            {
                using NeuronType = typename std::decay_t<decltype(population)>::PopulationNeuronType;
                normalize_neurons<NeuronType, DataGetter, DataSetter>(
                    const_cast<knp::core::Population<NeuronType> &>(population), DataGetter<NeuronType>(),
                    DataSetter<NeuronType>(), corrector);
            },
            population_variant);
    }
}


/**
 * @brief Normalize parameters of all neurons in a network and copy them to a new `Network` object.
 * @tparam DataGetter type of the class containing method to get parameter value.
 * @tparam DataSetter type of the class containing method to set parameter value.
 * @tparam ValueCorrector type of the normalization function for neuron parameters.
 * @param network network whose population neuron parameters you want to normalize.
 * @param corrector function used to normalize neuron parameter.
 * @return copy of the network with normalized neurons.
 */
template <
    template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
knp::framework::Network normalize_neurons(const knp::framework::Network &network, const ValueCorrector &corrector)
{
    auto new_network = network;

    normalize_neurons<DataGetter, DataSetter>(new_network, corrector);

    return new_network;
}

}  // namespace knp::framework::normalization
