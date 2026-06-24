/**
 * @file lif.h
 * @brief LIF neuron type traits.
 * @kaspersky_support David P.
 * @date 29.04.2026
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

#include <cstdint>

#include "type_traits.h"


/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{

/**
 * @brief Type that represents a classic Leaky‑Integrate‑and‑Fire neuron.
 * 
 * @note This type is intended to be used only as a template argument. It does not contain any data members.
 */
struct LIFNeuron;


/**
 * @brief Structure for LIF neuron default values.
 */
template <>
struct default_values<LIFNeuron>
{
    /**
     * @brief The parameter defines the default value for `potential_` state of LIF neuron.
     */
    constexpr static float potential_ = 0;

    /**
     * @brief The parameter defines the default value for `potential_reset_value_` of LIF neuron.
     */
    constexpr static float potential_reset_value_ = 0;

    /**
     * @brief The parameter defines the default value for `activation_threshold_` of LIF neuron.
     */
    constexpr static float activation_threshold_ = 1;

    /**
     * @brief The parameter defines the default value for `leak_coefficient_` of LIF neuron.
     */
    constexpr static float leak_coefficient_ = 1;

    /**
     * @brief The parameter defines the default value for `refract_counter_` of LIF neuron.
     */
    constexpr static uint32_t refract_counter_ = 0;

    /**
     * @brief The parameter defines the default value for `refract_period_` of LIF neuron.
     */
    constexpr static uint32_t refract_period_ = 0;
};


/**
 * @brief Structure for LIF neuron parameters.
 */
template <>
struct neuron_parameters<LIFNeuron>
{
    /**
     * @brief Current membrane potential. When the potential exceeds `activation_threshold_`, a spike is emitted
     * and the potential is reset to `potential_reset_value_`.
     */
    float potential_ = default_values<LIFNeuron>::potential_;

    /**
     * @brief Value to which the potential is reset after a spike.
     */
    float potential_reset_value_ = default_values<LIFNeuron>::potential_reset_value_;

    /**
     * @brief Activation threshold that triggers a spike.
     */
    float activation_threshold_ = default_values<LIFNeuron>::activation_threshold_;

    /**
     * @brief Leak coefficient applied on each pre-impact step. It multiplies the potential to model passive decay.
     */
    float leak_coefficient_ = default_values<LIFNeuron>::leak_coefficient_;

    /**
     * @brief Refractory counter. After a spike the counter is set to `refract_period_` and decremented each step.
     * Incoming impacts are ignored while the counter is greater than zero. 
     */
    uint32_t refract_counter_ = default_values<LIFNeuron>::refract_counter_;

    /**
     * @brief Refractory period.
     */
    uint32_t refract_period_ = default_values<LIFNeuron>::refract_period_;
};

}  // namespace knp::neuron_traits
