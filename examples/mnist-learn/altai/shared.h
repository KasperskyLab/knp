/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support D. Postnikov
 * @date 28.07.2025
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

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/neuron-traits/all_traits.h>
#include <knp/synapse-traits/all_traits.h>

// Network hyperparameters. You may want to fine-tune these.

// Number of neurons reserved per a single digit.
constexpr size_t neurons_per_column = 20;

// Ten possible digits, one column for each one.
constexpr size_t classes_amount = 10;

constexpr float state_increment_factor = 1.f / 255;

// Number of pixels in width for a single MNIST image.
constexpr size_t input_size_width = 28;

// Number of pixels in height for a single MNIST image.
constexpr size_t input_size_height = 28;

constexpr size_t images_amount_to_train = 10000;
constexpr size_t images_amount_for_inference = 2000;

constexpr size_t active_steps = 10;
constexpr size_t steps_per_image = 15;

/// How many subnetworks to use.
constexpr size_t num_subnetworks = 1;

// Number of pixels for a single MNIST image.
constexpr size_t input_size = input_size_width * input_size_height;
