/**
 * @file hyperparameters.h
 * @brief Network hyperparameters for AltAI neuron model.
 * @details These parameters are specifically tuned for the AltAI spiking neural network implementation and include both 
 * standard neural parameters and model-specific scaling factors for fixed-point arithmetic optimization.
 * Some parameters are multiplied by 1000 because the AltAI model uses fixed-point arithmetic instead of floating-point 
 * operations for computational efficiency. This scaling allows the model to operate effectively with integer-based 
 * calculations while maintaining necessary precision for learning and spiking dynamics.
 * @kaspersky_support D. Postnikov
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
 * @brief Number of neurons reserved per single digit class in the input population.
 * 
 * @details This parameter determines the columnar organization of the input population, with 20 neurons allocated for
 * each of the 10 digit classes (200 total input neurons). Each neuron in a column corresponds to a specific feature 
 * of the digit class.
 */
constexpr size_t neurons_per_column = 20;


/**
 * @brief the threshold value of neuron potential, after exceeding which a positive spike can be emitted.
 */
constexpr uint16_t activation_threshold = 8531;
/**
 * @brief The constant leakage of the neuron potential.
 */
constexpr int16_t potential_leak = 0;
/**
 * @brief The threshold value of neuron potential, below which a negative spike can be emitted
 */
constexpr uint16_t negative_activation_threshold = 0;
/**
 * @brief A reset value of the neuron potential after one of the thresholds has been exceeded.
 */
constexpr uint16_t potential_reset_value = 0;
/**
 * @brief Time parameter for dopamine plasticity.
 */
constexpr uint32_t dopamine_plasticity_time = 10;
/**
 * @brief Time between spikes in the ISI period.
 */
constexpr uint32_t isi_max = 10;
/**
 * @brief Hebbian plasticity value.
 */
constexpr float d_h = -0.1765261f * 1000;
/**
 * @brief Stability fluctuation value.
 */
constexpr float stability_change_parameter = 0.0497573 / 1000;
/**
 * @brief Number of silent synapses.
 */
constexpr uint32_t resource_drain_coefficient = 27;
/**
 * @brief Synapse sum threshold coefficient.
 */ 
constexpr float synapse_sum_threshold_coefficient = 0.217654;
/**
 * @brief Dopamine plasticity period for raster to input synapses.
 */
constexpr uint32_t raster_to_input_synapse_dopamine_plasticity_period = 10;
/**
 * @brief Minimum synaptic weight for raster to input projections.
 */
constexpr float raster_to_input_synapse_w_min = -0.253122 * 1000;
/**
 * @brief Maximum synaptic weight for raster to input projections.
 */
constexpr float raster_to_input_synapse_w_max = 0.0923957 * 1000;
