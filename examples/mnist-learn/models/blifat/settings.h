/**
 * @file shared.h
 * @brief Settings for BLIFAT network.
 * @kaspersky_support A. Vartenkov
 * @date 28.07.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include <settings.h>

#include <cstdint>


// Network hyperparameters. You may want to fine-tune these.
constexpr float default_threshold = 8.571F;
constexpr float min_synaptic_weight = -0.253;
constexpr float max_synaptic_weight = 0.0924;
constexpr float base_weight_value = 0.000F;
constexpr int neuron_dopamine_period = 10;
constexpr int synapse_dopamine_period = 10;
constexpr float input_neuron_potential_decay = 1.0 - 1.0 / 3.0;
constexpr float dopamine_parameter = 0.042F;
constexpr float hebbian_plasticity = -0.177;
constexpr float threshold_weight_coeff = 0.218F;
// Time between spikes in the ISI period.
constexpr uint32_t isi_max = 10;
// Minimum potential value.
constexpr double min_potential = 0;
// Defines stability fluctuation value.
constexpr float stability_change_parameter = 0.05F;
// Defines the number of silent synapses.
constexpr uint32_t resource_drain_coefficient = 27;
// Random number in range [0,stochastic_stimulation) that is added each step to the potential.
constexpr float stochastic_stimulation = 2.212;


// Number of neurons reserved per a single digit.
constexpr size_t neurons_per_column = 20;

// All columns are a part of the same population.
constexpr size_t num_input_neurons = neurons_per_column * classes_amount;

// How many subnetworks to use.
constexpr int num_subnetworks = 1;
