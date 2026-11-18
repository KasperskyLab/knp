/**
 * @file population.cpp
 * @brief Population class implementation.
 * @kaspersky_support Artiom N.
 * @date 11.01.2023
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


#include <knp/core/population.h>

#include <spdlog/spdlog.h>


namespace knp::core
{
template <typename NeuronType>
Population<NeuronType>::Population(Population<NeuronType>::NeuronGenerator generator, size_t neurons_count)  //!OCLINT
{
    SPDLOG_DEBUG(
        "Creating population with UID = {} and number of neurons = {}...", std::string(get_uid()), neurons_count);
    add_neurons(generator, neurons_count);
}


template <typename NeuronType>
Population<NeuronType>::Population(
    const UID &uid, Population<NeuronType>::NeuronGenerator generator,  //!OCLINT
    size_t neurons_count)                                               //!OCLINT
    : base_{uid}
{
    SPDLOG_DEBUG(
        "Creating population with UID = {} and number of neurons = {}...", std::string(get_uid()), neurons_count);
    add_neurons(generator, neurons_count);
}


#define INSTANCE_POPULATIONS(n, template_for_instance, neuron_type) \
    template class knp::core::Population<knp::neuron_traits::neuron_type>;

// cppcheck-suppress unknownMacro
BOOST_PP_SEQ_FOR_EACH(INSTANCE_POPULATIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_NEURONS))

}  // namespace knp::core

#define INSTANCE_ASTROCYTE_POPULATIONS(n, _, astrocyte_type) \
    template class knp::core::Population<knp::astrocyte_traits::astrocyte_type>;

BOOST_PP_SEQ_FOR_EACH(INSTANCE_ASTROCYTE_POPULATIONS, , BOOST_PP_VARIADIC_TO_SEQ(ALL_ASTROCYTES))
