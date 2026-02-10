/**
 * @file network_functions.h
 * @brief AltAI specific network functions.
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

// cppcheck-suppress missingInclude
#include "dataset.h"
// cppcheck-suppress missingInclude
#include "models/network_functions.h"


/// Specification of network construction for AltAILIF neuron.
template <>
AnnotatedNetwork construct_network<knp::neuron_traits::AltAILIF>(const ModelDescription& model_desc);


/// Specification of network finalization for AltAILIF neuron.
template <>
void finalize_network<knp::neuron_traits::AltAILIF>(AnnotatedNetwork& network, const ModelDescription& model_desc);


/// Specification of training labels spikes generator for AltAILIF neuron.
template <>
std::function<knp::core::messaging::SpikeData(knp::core::Step)>
make_training_labels_spikes_generator<knp::neuron_traits::AltAILIF>(const Dataset& dataset);
