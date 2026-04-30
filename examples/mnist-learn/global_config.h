/**
 * @file global_config.h
 * @brief Global configuration.
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

#include <cstddef>


/**
 * @brief Number of classification classes in the dataset.
 * 
 * @note For MNIST dataset, there are 10 classes representing digits 0 through 9.
 */
constexpr size_t classes_amount = 10;

/**
 * @brief Input image size in pixels.
 * 
 * @details MNIST images are 28×28 pixels, resulting in 784 total pixels per image.
 */
constexpr size_t input_size = 28 * 28;

/**
 * @brief Total number of simulation steps per input image.
 * 
 * @details Each image is processed over 15 simulation steps to generate spike patterns.
 */
constexpr size_t steps_per_image = 15;

/**
 * @brief Number of active simulation steps for spike transmission.
 * 
 * @details Out of total @p steps_per_image, only 10 steps are used for actual spike transmission to the neural network.
 */
constexpr size_t active_steps = 10;

/**
 * @brief Number of winners in Winner-Take-All (WTA) mechanism.
 * 
 * @details Only one neuron wins in the WTA competition during each time step.
 */
constexpr size_t wta_winners_amount = 1;

/**
 * @brief Scaling factor for converting pixel intensities to spike rates.
 * 
 * @details Pixel values range from 0-255, this factor scales them to appropriate spike generation rates.
 */
constexpr float state_increment_factor = 1 / 255.f;

/**
 * @brief Logging period for aggregated spike data.
 * 
 * @details Every 4,000 simulation steps, aggregated spike information is written to file when logging is enabled.
 */
constexpr size_t aggregated_spikes_logging_period = 4e3;

/**
 * @brief Logging period for projection weights.
 * 
 * @details Every 100,000 simulation steps, synaptic weights are written to file when logging is enabled.
 */
constexpr size_t projection_weights_logging_period = 1e5;
