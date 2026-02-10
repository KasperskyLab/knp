/**
 * @file global_parameters.h
 * @brief Global parameters.
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

/// Amount of classes, its 10 because this is MNIST, we are predicting numbers from 0 to 9.
constexpr size_t classes_amount = 10;

/// Size of an image, its 28 by 28 pixels.
constexpr size_t input_size = 28 * 28;

/// Amount of steps for each image.
constexpr size_t steps_per_image = 15;

/// Amount of steps out of steps_per_image, during which we are sending image into network in spikes form.
constexpr size_t active_steps = 10;

/// Amount of winners in WTA(winner takes all).
constexpr size_t wta_winners_amount = 1;

/// This is used for transforming images into spikes form.
constexpr float state_increment_factor = 1 / 255.f;

/// Each aggregated_spikes_logging_period steps, all aggregated spikes will be written to a file, it logging is enabled.
constexpr size_t aggregated_spikes_logging_period = 4e3;

/// Each projection_weights_logging_period steps, all projections weights will be written to a file, it logging is
/// enabled.
constexpr size_t projection_weights_logging_period = 1e5;
