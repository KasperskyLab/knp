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

#include <annotated_network.h>
#include <dataset.h>
#include <model_desc.h>


template <typename Neuron>
AnnotatedNetwork construct_network(const ModelDescription& model_desc)
{
    throw std::runtime_error("Not supported neuron type.");
}


template <typename Neuron>
void finalize_network(AnnotatedNetwork& network, const ModelDescription& model_desc)
{
    throw std::runtime_error("Not supported neuron type.");
}


template <typename Neuron>
std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_labels_spikes_generator(
    const Dataset& dataset)
{
    throw std::runtime_error("Not supported neuron type.");
}
