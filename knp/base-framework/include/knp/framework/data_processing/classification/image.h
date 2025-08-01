/**
 * @file image.h
 * @brief Processing of dataset of images.
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

#include <vector>

#include "dataset.h"


namespace knp::framework::data_processing::classification::images
{

/**
 * @brief A class that represents processed dataset of images.
 */
class KNP_DECLSPEC Dataset : public classification::Dataset
{
public:
    /**
     * @brief Create data pairs from labels and images, that are converted to spikes form.
     * @param images_stream Stream of raw images.
     * @param labels_stream Stream of labels.
     * @param training_amount Amount of images you want to train model on.
     * @param classes_amount Amount of classes.
     * @param image_size Size of an images.
     * @param steps_per_image Amount of steps required to send image in spikes form to a model.
     * @param image_to_spikes Converter or raw image data to spikes form.
     */
    void process_labels_and_images(
        std::istream &images_stream, std::istream &labels_stream, size_t training_amount, size_t classes_amount,
        size_t image_size, size_t steps_per_image,
        std::function<Frame(std::vector<uint8_t> const &)> const &image_to_spikes);

    /**
     * @brief Make generator of spikes, from training labels, for channel.
     * @return A functor for generating spikes from dataset.
     */
    [[nodiscard]] std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_labels_generator()
        const;

    /**
     * @brief Make generator of spikes, from training images in form of spikes, for channel.
     * @return A functor for generating spikes from dataset.
     */
    [[nodiscard]] std::function<knp::core::messaging::SpikeData(knp::core::Step)>
    make_training_images_spikes_generator() const;

    /**
     * @brief Make generator of spikes, from inference images in form of spikes, for channel.
     * @return A functor for generating spikes from dataset.
     */
    [[nodiscard]] std::function<knp::core::messaging::SpikeData(knp::core::Step)>
    make_inference_images_spikes_generator() const;

    /**
     * @brief Create a incrementing image to spikes converter
     * @detail Spikes will be sent for active_steps steps, and spikes wont be sent for steps_per_image-active_steps
     * steps. This converter considered incrementing because it will add state_increment_factor * image_pixel to states,
     * and when value is greater than one, this considered a spike.
     * @param active_steps Amount of active steps, active steps are steps when spikes being sent, must be <
     * steps_per_image.
     * @param state_increment_factor How much to increment to spike accumulator.
     * @return A functor that converts image raw data to spikes.
     */
    [[nodiscard]] std::function<Frame(std::vector<uint8_t> const &)> make_incrementing_image_to_spikes_converter(
        size_t active_steps, float state_increment_factor) const;

    /**
     * @brief Get image size.
     * @ret Image size.
     */
    [[nodiscard]] size_t get_image_size() const { return image_size_; }

protected:
    /**
     * @brief Total image size.
     */
    size_t image_size_ = 0;
};


}  // namespace knp::framework::data_processing::classification::images
