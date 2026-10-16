/**
 * @file slicing_generators.h
 * @brief Utilities that can be used for neuron population upcasting.
 * @kaspersky_support A. Vartenkov
 * @date 17.02.2026
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

#include <knp/core/population.h>
#include <knp/neuron-traits/blifat.h>


/**
 * @brief Namespace that contains utilities for generating neuron parameters.
 */
namespace knp::framework::population::neurons_generators
{
/**
 * @brief Slice a derived neuron parameter set to its base neuron counterpart.
 *
 * @tparam BaseNeuron base neuron type (simpler).
 * @tparam DerivedNeuron neuron type that publicly derives from the base type.
 *
 * @param derived_params parameters of the derived neuron that should be sliced.
 *
 * @return parameters of the base neuron obtained by static casting.
 *
 * @details @p BaseNeuron is a simple neuron type that contains only the core parameters needed for the
 * simulation. @p DerivedNeuron is a more complex neuron that inherits from the base neuron structure and
 * adds extra fields. For example, @ref knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron publicly derives
 * from knp::neuron_traits::BLIFATNeuron. Consequently,
 * `neuron_parameters<SynapticResourceSTDPBLIFATNeuron>` can be sliced to `neuron_parameters<BLIFATNeuron>`
 * object, discarding the extra fields (`dopamine_value_`, `is_being_forced_`, and `resource_`).
 *
 * @note This generator is useful for creating a non-trainable model from a trainable one, or when
 * moving a model to a backend that does not support training.
 */
template <class BaseNeuron, class DerivedNeuron>
neuron_traits::neuron_parameters<BaseNeuron> slice_to_base_neuron(
    const neuron_traits::neuron_parameters<DerivedNeuron> &derived_params)
{
    static_assert(std::is_base_of_v<
                  neuron_traits::neuron_parameters<BaseNeuron>, neuron_traits::neuron_parameters<DerivedNeuron>>);
    return static_cast<neuron_traits::neuron_parameters<BaseNeuron>>(derived_params);
}


/**
 * @brief Create a neuron generator that produces base neuron parameters from a population of derived
 * neurons.
 *
 * @tparam BaseNeuron base neuron type (simpler).
 * @tparam DerivedNeuron neuron type that publicly derives from the base type.
 *
 * @param derived_population population containing the derived neurons.
 *
 * @details @p BaseNeuron is a simple neuron type that contains only the core parameters needed for the
 * simulation. @p DerivedNeuron is a more complex neuron that inherits from the base neuron structure and
 * adds extra fields. For example, @ref knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron publicly derives
 * from knp::neuron_traits::BLIFATNeuron. Consequently,
 * `neuron_parameters<SynapticResourceSTDPBLIFATNeuron>` can be sliced to `neuron_parameters<BLIFATNeuron>`
 * object, discarding the extra fields (`dopamine_value_`, `is_being_forced_`, and `resource_`).
 *
 * @return a generator (`NeuronGenerator`) that yields base neuron parameters on each step.
 */
template <class BaseNeuron, class DerivedNeuron>
[[nodiscard]] typename knp::core::Population<BaseNeuron>::NeuronGenerator make_slicing_neuron_generator(
    const knp::core::Population<DerivedNeuron> &derived_population)
{
    using BaseParams = knp::neuron_traits::neuron_parameters<BaseNeuron>;
    using BaseGenerator = typename knp::core::Population<BaseNeuron>::NeuronGenerator;
    BaseGenerator generator = [derived_population](size_t step) -> std::optional<BaseParams>
    {
        if (step >= derived_population.size()) return std::optional<BaseParams>{};
        return slice_to_base_neuron<BaseNeuron>(derived_population[step]);
    };
    return generator;
}


/**
 * @brief Convert a population of derived neurons into a population of base neurons.
 *
 * @tparam BaseNeuron base neuron type (simpler).
 * @tparam DerivedNeuron neuron type that publicly derives from the base type.

 * @param input_population the population to be converted.
 *
 * @return a new `Population<BaseNeuron>` that contains the sliced parameters of the original population.
 *
 * @details @p BaseNeuron is a simple neuron type that contains only the core parameters needed for the
 * simulation. @p DerivedNeuron is a more complex neuron that inherits from the base neuron structure and
 * adds extra fields. For example, @ref knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron publicly derives
 * from knp::neuron_traits::BLIFATNeuron. Consequently,
 * `neuron_parameters<SynapticResourceSTDPBLIFATNeuron>` can be sliced to `neuron_parameters<BLIFATNeuron>`
 * object, discarding the extra fields (`dopamine_value_`, `is_being_forced_`, and `resource_`).
 *
 * @note The function builds a new population using the same UID as the source and the original size.
 * The resulting population can be passed to backends that only understand the base neuron type.
 */
template <class BaseNeuron, class DerivedNeuron>
[[nodiscard]] knp::core::Population<BaseNeuron> upcast_population(
    const knp::core::Population<DerivedNeuron> &input_population)
{
    knp::core::Population<BaseNeuron> result{
        input_population.get_uid(), make_slicing_neuron_generator<BaseNeuron, DerivedNeuron>(input_population),
        input_population.size()};
    return result;
}
}  // namespace knp::framework::population::neurons_generators
