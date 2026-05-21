/**
 * @file dataset.h
 * @brief Process dataset.
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

#include <knp/framework/data_processing/classification/image.h>

#include "model_desc.h"


/**
 * @brief Short name for dataset class.
 */ 
using Dataset = knp::framework::data_processing::classification::images::Dataset;

/**
 * @brief Process MNIST dataset for neural network training and inference.
 * 
 * @details This function reads raw MNIST dataset files (images and labels), loads them into memory, converts the data 
 * into spike representations suitable for neural networks, and splits the dataset into training and inference portions 
 * according to the model configuration.
 * 
 * @param model_desc model description.
 * 
 * @return Processed dataset object ready for training and inference operations.
 * 
 * @throws std::runtime_error if required dataset files are not found or cannot be read.
 */
Dataset process_dataset(ModelDescription const& model_desc);
