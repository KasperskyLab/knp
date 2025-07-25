/**
 * @file image_classification.h
 * @brief Processing of dataset
 * @kaspersky_support D. Postnikov
 * @date 14.07.2025
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

#pragma once

#include <knp/core/core.h>
#include <knp/core/impexp.h>
#include <knp/core/messaging/messaging.h>

#include <utility>
#include <vector>

#include "classification_dataset.h"


namespace knp::framework::data_processing::classification::images
{

/**
 * @brief A class that represents processed dataset.
 */
struct Dataset : classification::Dataset
{
    /**
     * @brief Total image size.
     */
    size_t image_size_;
};


/**
 * @brief Create a incrementing image to spikes converter, its gonna send spikes for active_steps steps, and it will not
 * send any for steps_per_image-active_steps steps.
 * @detail This converter considered incrementing because it will add state_increment_factor * image_pixel to some
 * table, and when sum is greater than 1, it will add spike
 * @param steps_per_image amount of steps required for a single image to be passed to model in form of spikes.
 * @param active_steps amount of active steps, active steps are steps when spikes being sent, must me < steps_per_image.
 * @param image_size total image size.
 * @param state_increment_factor how much to increment to spike accumulator.
 * @param states a vector of states, should be filled with zeroes and size of image_size, should have same lifetime as
 * function call expression lifetime.
 * @return A functor that converts image raw data to spikes.
 */
KNP_DECLSPEC std::function<std::vector<bool>(std::vector<uint8_t> const &)> make_incrementing_image_to_spikes_converter(
    size_t steps_per_image, size_t active_steps, size_t image_size, float state_increment_factor,
    std::vector<float> &&states);


/**
 * @brief Process data from dataset, and save it in form so it can be used for training and inference. Note that dataset
 * should be randomized beforehand, this function does not include any randomization.
 * @param images_stream stream of raw images.
 * @param labels_stream stream of labels to images.
 * @param training_amount amount of images model should be trained on.
 * @param dataset_split a float between 0 and 1, represents dataset split on training and inference data, for example
 * dataset_split of 0.8 would cut dataset so 80% of data is used for training, and other data used for inference.
 * @param image_size total image size.
 * @param steps_per_image amount of steps used to send image in form of spikes to model.
 * @param image_to_spikes function to convert raw image data to spikes form.
 * @return Processed dataset.
 */
KNP_DECLSPEC Dataset process_data(
    std::istream &images_stream, std::istream &labels_stream, size_t training_amount, float dataset_split,
    size_t image_size, size_t steps_per_image,
    std::function<std::vector<bool>(std::vector<uint8_t> const &)> const &image_to_spikes);


/**
 * @brief Make generator of spikes, from training labels, for channel.
 * @param dataset dataset.
 * @return A functor for generating spikes from dataset.
 */
KNP_DECLSPEC std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_labels_generator(
    const Dataset &dataset);


/**
 * @brief Make generator of spikes, from training images in form of spikes, for channel.
 * @param dataset dataset.
 * @return A functor for generating spikes from dataset.
 */
KNP_DECLSPEC std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_images_spikes_generator(
    const Dataset &dataset);


/**
 * @brief Make generator of spikes, from inference images in form of spikes, for channel.
 * @param dataset dataset.
 * @return A functor for generating spikes from dataset.
 */
KNP_DECLSPEC std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_inference_images_spikes_generator(
    const Dataset &dataset);

}  // namespace knp::framework::data_processing::classification::images
