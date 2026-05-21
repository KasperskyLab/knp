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

#include <memory>

// cppcheck-suppress missingInclude
#include "annotated_network.h"
// cppcheck-suppress missingInclude
#include "dataset.h"
// cppcheck-suppress missingInclude
#include "model_desc.h"


/**
 * @brief Generic network construction function with fallback error handling.
 * 
 * @details This template provides the generic interface for network construction across all supported neuron 
 * models. When a specific neuron model specialization is not implemented, this generic version throws a runtime 
 * error indicating the neuron type is not supported. Each supported neuron model should implement a template
 * specialization for proper network construction.
 * 
 * @tparam Neuron neuron type.
 * 
 * @param model_desc model description containing configuration parameters.
 * 
 * @return constructed `AnnotatedNetwork` for the specified neuron type.
 * 
 * @throws `std::runtime_error` if the neuron type is not supported.
 */
template <typename Neuron>
AnnotatedNetwork construct_network(const ModelDescription& model_desc)
{
    throw std::runtime_error("Not supported neuron type.");
}


/**
 * @brief Generic network preparation function for inference with fallback error handling.
 * 
 * @details This template provides the generic interface for preparing networks for inference operations across all 
 * supported neuron models. When a specific neuron model specialization is not implemented, this generic version 
 * throws a runtime error indicating the neuron type is not supported. Each supported neuron model should implement 
 * a template specialization for proper inference preparation.
 * 
 * @tparam Neuron neuron type.
 * 
 * @param backend shared pointer to the computational backend for inference execution.
 * @param model_desc model description containing configuration parameters and paths.
 * @param network annotated network structure to prepare for inference.
 * 
 * @throws `std::runtime_error` if the neuron type is not supported.
 */
template <typename Neuron>
void prepare_network_for_inference(
    const std::shared_ptr<knp::core::Backend>& backend, const ModelDescription& model_desc, AnnotatedNetwork& network)
{
    throw std::runtime_error("Not supported neuron type.");
}


/**
 * @brief Generic training labels spike generator function with fallback error handling.
 * 
 * @details This template provides the generic interface for creating training label spike generators across all supported 
 * neuron models. When a specific neuron model specialization is not implemented, this generic version throws a runtime error
 * indicating the neuron type is not supported. Each supported neuron model should implement a template specialization for 
 * proper label spike generation.
 * 
 * @tparam Neuron neuron type.
 * 
 * @param dataset dataset containing training labels and configuration.
 * 
 * @return callable function that generates spike data for training labels.
 * 
 * @throws `std::runtime_error` if the neuron type is not supported.
 */
template <typename Neuron>
std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_labels_spikes_generator(
    const Dataset& dataset)
{
    throw std::runtime_error("Not supported neuron type.");
}
