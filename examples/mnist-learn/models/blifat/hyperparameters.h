/**
 * @file hyperparameters.h
 * @brief BLIFAT neuron model hyperparameters.
 * @details This header file defines all hyperparameters for the BLIFAT neuron model implementation.
 * The parameters are carefully tuned for spiking neural network learning with STDP mechanisms
 * and are organized into logical categories for better understanding and maintenance.
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

#include <cstdint>

// cppcheck-suppress missingInclude
#include "global_config.h"


/**
 * @brief Default activation threshold for BLIFAT neurons.
 */
constexpr float default_threshold = 8.571F;
/**
 * @brief Minimum synaptic weight value.
 */
constexpr float min_synaptic_weight = -0.253;
/**
 * @brief Maximum synaptic weight value
 */
constexpr float max_synaptic_weight = 0.0924;
/**
 * @brief Base weight value
 */
constexpr float base_weight_value = 0.000F;
/**
 * @brief Dopamine plasticity period for neuron-level learning.
 */
constexpr uint32_t neuron_dopamine_period = 10;
/**
 * @brief Dopamine plasticity period for synapse-level learning.
 */
constexpr uint32_t synapse_dopamine_period = 10;
/**
 * @brief A time constant during which the input neuron potential tends to zero.
 */
constexpr float input_neuron_potential_decay = 1.0 - 1.0 / 3.0;
/**
 * @brief Dopamine modulation parameter for learning signals.
 */
constexpr float dopamine_parameter = 0.042F;
/**
 * @brief Hebbian plasticity for STDP learning mechanisms.
 */
constexpr float hebbian_plasticity = -0.177;
/**
 * @brief Threshold weight coefficient for synaptic sum calculations.
 */
constexpr float threshold_weight_coeff = 0.218F;
/**
 * @brief Time between spikes in the ISI period.
 */
constexpr uint32_t isi_max = 10;
/**
 * @brief A value to which membrane potential tends.
 */
constexpr double min_potential = 0;
/**
 * @brief The stability fluctuation value.
 */
constexpr float stability_change_parameter = 0.05F;
/**
 * @brief The number of silent synapses.
 */
constexpr uint32_t resource_drain_coefficient = 27;
/**
 * @brief Random number in range [0,stochastic_stimulation) that is added to the potential every tick.
 */
constexpr float stochastic_stimulation = 2.212;


/**
 * @brief Number of neurons reserved per digit class in input population.
 * 
 * @details This parameter determines the columnar organization of the input population, with 20 neurons 
 * allocated for each of the 10 digit classes (200 total input neurons). Each neuron in a column corresponds 
 * to a specific feature of the digit class.
 */
constexpr size_t neurons_per_column = 20;

/**
 * @brief Total number of input neurons in the network.
 * 
 * @details Calculated as neurons_per_column × classes_amount, providing a direct mapping from digit classes 
 * to input neuron populations.
 */
constexpr size_t num_input_neurons = neurons_per_column * classes_amount;

/**
 * @brief Number of independent subnetworks to use.
 * 
 * @details This parameter controls the multi-subnetwork architecture of the BLIFAT model. Setting to 1 creates
 *  a single network, while higher values enable distributed processing across multiple network instances.
 */
constexpr size_t num_subnetworks = 1;
