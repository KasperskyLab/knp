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
constexpr float default_threshold = 8.571F;
constexpr float min_synaptic_weight = -0.7;
constexpr float max_synaptic_weight = 0.864249F;
constexpr float base_weight_value = 0.000F;
constexpr int neuron_dopamine_period = 10;
constexpr int synapse_dopamine_period = 10;
constexpr float l_neuron_potential_decay = 1.0 - 1.0 / 3.0;
constexpr float dopamine_parameter = 0.042F;
constexpr float dopamine_value = dopamine_parameter;
constexpr float threshold_weight_coeff = 0.023817F;

//
// Network geometry.
//

// Number of neurons reserved per a single digit.
constexpr size_t neurons_per_column = 15;

// Ten possible digits, one column per each.
constexpr size_t num_possible_labels = 10;

// All columns are a part of the same population.
constexpr size_t num_input_neurons = neurons_per_column * num_possible_labels;

// Number of pixels in width for a single MNIST image.
constexpr size_t input_size_width = 28;

// Number of pixels in height for a single MNIST image.
constexpr size_t input_size_height = 28;

// Number of pixels for a single MNIST image.
constexpr size_t input_size = input_size_width * input_size_height;

// Dense input projection from 28 * 28 image to population of 150 neurons.
constexpr size_t input_projection_size = input_size * num_input_neurons;
