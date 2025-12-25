/**
 * @file delta.h
 * @brief Delta synapse type traits.
 * @kaspersky_support Andrey V.
 * @date 26.01.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <cinttypes>
#include <numeric>

#include "output_types.h"
#include "type_traits.h"

/**
 * @brief Namespace for synapse traits.
 */
namespace knp::synapse_traits
{
/**
 * @brief Delta synapse.
 * @note Use as a template parameter only.
 */
struct DeltaSynapse;


/**
 * @brief Default values for delta synapse parameters.
 */
template <>
struct default_values<DeltaSynapse>
{
    /**
     * @brief Synaptic weight default value.
     */
    constexpr static float weight_ = 0.0F;

    /**
     * @brief Synaptic delay default value.
     * @note Value of `1` is the least delay possible.
     */
    constexpr static uint32_t delay_ = 1;

    /**
     * @brief Synapse type default value.
     */
    constexpr static OutputType output_type_ = OutputType::EXCITATORY;
};


/**
 * @brief Structure for delta synapse parameters.
 */
template <>
struct synapse_parameters<DeltaSynapse>
{
    /**
     * @brief Synaptic weight.
     */
    float weight_ = default_values<DeltaSynapse>::weight_;

    /**
     * @brief Synaptic delay.
     * @details Delay of `N` means that a spike sent on step `X` will be received on step `X + N`.
     */
    uint32_t delay_ = default_values<DeltaSynapse>::delay_;

    /**
     * @brief Synapse type.
     * @details Various types have different influences on neuron parameters, as defined by neuron
     * function.
     */
    knp::synapse_traits::OutputType output_type_ = default_values<DeltaSynapse>::output_type_;
};

}  // namespace knp::synapse_traits
