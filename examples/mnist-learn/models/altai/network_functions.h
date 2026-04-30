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

#include <memory>

// cppcheck-suppress missingInclude
#include "dataset.h"
// cppcheck-suppress missingInclude
#include "models/network_functions.h"


/**
 * @brief Specialized network construction function for AltAI neuron model.
 * 
 * @details This template specialization creates the complete network architecture specifically designed for the AltAI
 *  neuron model. It constructs populations, synapses, and connections tailored to the AltAI neuron's unique properties
 *  and requirements.
 * 
 * @param model_desc model description containing configuration parameters.
 * 
 * @return `AnnotatedNetwork` containing the fully constructed AltAI network.
 */
template <>
AnnotatedNetwork construct_network<knp::neuron_traits::AltAILIF>(const ModelDescription& model_desc);


/**
 * @brief Specialized network preparation function for inference with AltAI neuron model.
 * 
 * @details This template specialization prepares a trained AltAI network for inference operations. It configures the 
 * network state, sets up necessary annotations, and ensures proper execution for testing and validation phases.
 * 
 * @param backend shared pointer to the computational backend for inference execution.
 * @param model_desc model description containing configuration parameters and paths.
 * @param network annotated network structure to prepare for inference.
 */
template <>
void prepare_network_for_inference<knp::neuron_traits::AltAILIF>(
    const std::shared_ptr<knp::core::Backend>& backend, const ModelDescription& model_desc, AnnotatedNetwork& network);


/**
 * @brief Specialized training labels spike generator for AltAI neuron model.
 * 
 * @details This template specialization creates a spike generator specifically designed for generating label spikes 
 * during training with the AltAI neuron model. It converts label information from the dataset into appropriate spike 
 * patterns for supervised learning.
 * 
 * @param dataset dataset containing training labels and configuration.
 * @return callable function that generates spike data for training labels.
 */
template <>
std::function<knp::core::messaging::SpikeData(knp::core::Step)>
make_training_labels_spikes_generator<knp::neuron_traits::AltAILIF>(const Dataset& dataset);
