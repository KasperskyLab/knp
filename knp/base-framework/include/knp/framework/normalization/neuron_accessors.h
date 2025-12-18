/**
 * @file neuron_accessors.h
 * @brief Neuron parameter accessor functions.
 * @kaspersky_support Artiom N.
 * @date 30.07.2025
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

#include <knp/core/population.h>
#include <knp/framework/normalization/normalizers.h>
#include <knp/neuron-traits/all_traits.h>


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{

/**
 * @brief The `PotentialAccessor` class is a definition of an interface that provides access to neuron potential parameter (potential_). 
 * @tparam NeuronType type of the neuron to access.
 */
template <typename NeuronType>
struct PotentialAccessor
{
    /**
     * @brief Type of neuron parameters.
     */
    using NeuronParametersType = typename knp::core::Population<NeuronType>::NeuronParameters;
    /**
     * @brief Neuron potential type.
     */
    using PotentialType = decltype(NeuronParametersType::potential_);

    /**
     * @brief Get neuron potential.
     * @param neuron_params neuron parameters.
     * @return value of the neuron potential.
     */
    PotentialType operator()(const NeuronParametersType &neuron_params) const { return neuron_params.potential_; }

    /**
     * @brief Correct the neuron potential value.
     * @param neuron_params neuron parameters.
     * @param potential new neuron potential value to set.
     */
    void operator()(NeuronParametersType &neuron_params, const PotentialType &potential) const
    {
        neuron_params.potential_ = potential;
    }
};

}  // namespace knp::framework::normalization
