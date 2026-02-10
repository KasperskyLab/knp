/**
 * @file hyperparameters.h
 * @brief AltAI network hyperparameters.
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

#include <global_config.h>

#include <cstdint>


/// Number of neurons reserved per a single digit.
constexpr size_t neurons_per_column = 20;

// Network hyperparameters.
// Some of them are multiplied by 1000 because altai model is scaled up, so it can work without relying on floats for
// some operations.
/// Activation threshold.
constexpr uint16_t activation_threshold = 8531;
/// Potential leak.
constexpr int16_t potential_leak = static_cast<int16_t>(-(1.f - 1.f / 3.f) * 1000);
/// Negative activation threshold.
constexpr uint16_t negative_activation_threshold = 0;
/// Potential reset value.
constexpr uint16_t potential_reset_value = 0;
/// Dopamine plasticity time.
constexpr uint32_t dopamine_plasticity_time = 10;
/// Time between spikes in the ISI period.
constexpr uint32_t isi_max = 10;
/// Hebbian plasticity value.
constexpr float d_h = -0.1765261f * 1000;
/// Stability change parameter.
constexpr float stability_change_parameter = 0.0497573 / 1000;
/// Resource drain coefficient.
constexpr uint32_t resource_drain_coefficient = 27;
/// Synapse sum threshold coefficient.
constexpr float synapse_sum_threshold_coefficient = 0.217654;
/// Raster to input synapse dopamine plasticity period.
constexpr uint32_t raster_to_input_synapse_dopamine_plasticity_period = 10;
/// Raster to input synapse w min.
constexpr float raster_to_input_synapse_w_min = -0.253122 * 1000;
/// Raster to input synapse w max.
constexpr float raster_to_input_synapse_w_max = 0.0923957 * 1000;
