/**
 * @file evaluate_results.h
 * @brief Function for evaluating inference results.
 * @kaspersky_support D. Postnikov
 * @date 04.02.2026
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

#include <vector>

#include "dataset.h"


/**
 * @brief Evaluate inference results and display classification metrics.
 * 
 * @details This function processes spike data generated during inference operations, compares the results against ground truth labels 
 * from the dataset, and prints detailed classification performance metrics to standard output.
 * 
 * @param inference_spikes vector of spike messages generated during inference.
 * @param dataset dataset used for inference.
 */
void evaluate_results(const std::vector<knp::core::messaging::SpikeMessage>& inference_spikes, const Dataset& dataset);
