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
 * @brief namespace for neuron generators.
 */
namespace knp::framework::population::neurons_generators
{
/**
 * @brief Converts parameters of a derived neuron to its base neuron by slicing.
 * @tparam BaseNeuron base neuron type, simple.
 * @tparam DerivedNeuron derived neuron type, more complex.
 * @param derived_params neuron parameters to be sliced.
 * @return base neuron parameters.
 * @note Use this generator to create non-trainable model from a trainable one, or when moving a model to non-training
 * backend.
 */
template <class BaseNeuron, class DerivedNeuron>
neuron_traits::neuron_parameters<BaseNeuron> slice_to_base_neuron(
        const neuron_traits::neuron_parameters<DerivedNeuron> &derived_params)
{
    static_assert(std::is_base_of_v<
            neuron_traits::neuron_parameters<BaseNeuron>,
            neuron_traits::neuron_parameters<DerivedNeuron>>);
    return static_cast<neuron_traits::neuron_parameters<BaseNeuron>>(derived_params);
}


/**
 * @brief Make a generator of base neuron parameters from a derived population.
 * @tparam BaseNeuron base neuron type, simple.
 * @tparam DerivedNeuron derived neuron type, more complex.
 * @param derived_population a population of derived neurons.
 * @return a population of base neuron.
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
 * @brief Convert a more complex population of derived neurons to a more simple population of base neurons.
 * @param input_population the population to be converted.
 * @tparam BaseNeuron a simple neuron type.
 * @tparam DerivedNeuron neuron derived from a more simple type.
 * @return population of base neurons.
 */
template <class BaseNeuron, class DerivedNeuron>
[[nodiscard]] knp::core::Population<BaseNeuron> upcast_population(
        const knp::core::Population<DerivedNeuron> &input_population)
{
    knp::core::Population<BaseNeuron> result{input_population.get_uid(),
                                             make_slicing_neuron_generator<BaseNeuron, DerivedNeuron>
                                                     (input_population),
                                             input_population.size()};
    return result;
}
} // namespace knp::framework::population::neurons_generators
