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


using Dataset = knp::framework::data_processing::classification::images::Dataset;

/**
 * @brief Function for processing dataset.
 * @note Processing means reading actual files, saving them in memory, and splitting data into training and inference.
 * @param model_desc Model description.
 * @return Processed dataset.
 */
Dataset process_dataset(ModelDescription const& model_desc);
