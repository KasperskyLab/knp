/**
 * @file dataset.cpp
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

#include "dataset.h"

#include <fstream>

#include "global_config.h"


Dataset process_dataset(ModelDescription const& model_desc)
{
    // Check if required files exist.
    if (!std::filesystem::exists(model_desc.images_file_path_))
        throw std::runtime_error("Provided images file does not exists.");
    else if (!std::filesystem::exists(model_desc.labels_file_path_))
        throw std::runtime_error("Provided labels file does not exists.");

    // Create streams for reading images and labels.
    std::ifstream images_stream(model_desc.images_file_path_, std::ios::binary);
    std::ifstream labels_stream(model_desc.labels_file_path_, std::ios::in);

    Dataset dataset;
    // Process images and labels into spike representations.
    dataset.process_labels_and_images(
        images_stream, labels_stream, model_desc.train_images_amount_ + model_desc.inference_images_amount_,
        classes_amount, input_size, steps_per_image,
        dataset.make_incrementing_image_to_spikes_converter(active_steps, state_increment_factor));
    
    // Split dataset into training and inference portions.
    dataset.split(model_desc.train_images_amount_, model_desc.inference_images_amount_);

    // Print processing results and statistics.
    std::cout << "Processed dataset, training will last " << dataset.get_steps_amount_for_training()
              << " steps, inference " << dataset.get_steps_amount_for_inference() << " steps\n"
              << std::endl;

    return dataset;
}
