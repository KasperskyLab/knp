/**
 * @file network_functions.h
 * @brief Network functions for specific model types.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
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

#include <knp/neuron-traits/all_traits.h>

// cppcheck-suppress missingInclude
#include "annotated_network.h"
// cppcheck-suppress missingInclude
#include "dataset.h"
// cppcheck-suppress missingInclude
#include "model_desc.h"


/**
 * @brief If no neuron made a specialization for its type, throw exception.
 * @tparam Neuron Neuron type.
 * @param model_desc Model desciption.
 * @return Constructed network.
 */
template <typename Neuron>
AnnotatedNetwork construct_network(const ModelDescription& model_desc)
{
    throw std::runtime_error("Not supported neuron type.");
}


/**
 * @brief If no neuron made a specialization for its type, throw exception.
 * @tparam Neuron Neuron type.
 * @param network Annotated network.
 * @param model_desc Model description.
 */
template <typename Neuron>
void finalize_network(AnnotatedNetwork& network, const ModelDescription& model_desc)
{
    throw std::runtime_error("Not supported neuron type.");
}


/**
 * @brief If no neuron made a specialization for its type, throw exception.
 * @tparam Neuron Neuron type.
 * @param dataset Dataset.
 * @return Callable function on each step.
 */
template <typename Neuron>
std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_labels_spikes_generator(
    const Dataset& dataset)
{
    throw std::runtime_error("Not supported neuron type.");
}
