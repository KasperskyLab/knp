/**
 * @file altai_lif_population_impl.h
 * @kaspersky_support Vartenkov A.
 * @date 07.04.2025
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

#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>
#include <knp/neuron-traits/altai_lif.h>

#include <vector>

#include "lif_population_impl.h"


namespace knp::backends::cpu
{

// TODO: Maybe make this a .cpp and change library type from INTERFACE to STATIC later
template <>
void calculate_pre_input_state_lif<knp::neuron_traits::AltAILIF>(
    knp::core::Population<knp::neuron_traits::AltAILIF> &population)
{
    for (auto &neuron : population)
    {
        neuron.potential_ = std::round(neuron.potential_);

        neuron.potential_ = neuron.do_not_save_ ? static_cast<float>(neuron.potential_reset_value_) : neuron.potential_;
    }
}


void leak_potential(knp::core::Population<knp::neuron_traits::AltAILIF> &population)
{
    for (auto &neuron : population)
    {
        // -1 if leak_rev is true and potential < 0, 1 otherwise.
        const int sign = (neuron.leak_rev_ && neuron.potential_ < 0) ? -1 : 1;
        neuron.potential_ += neuron.potential_leak_ * sign;
    }
}


template <>
void process_inputs_lif<knp::neuron_traits::AltAILIF>(
    knp::core::Population<knp::neuron_traits::AltAILIF> &population,
    const std::vector<knp::core::messaging::SynapticImpactMessage> &messages)
{
    for (const auto &msg : messages)
    {
        for (const auto &impact : msg.impacts_)
        {
            population[impact.postsynaptic_neuron_index_].potential_ += impact.impact_value_;
        }
    }
    leak_potential(population);
}


template <>
knp::core::messaging::SpikeData calculate_spikes_lif<knp::neuron_traits::AltAILIF>(
    knp::core::Population<knp::neuron_traits::AltAILIF> &population)
{
    knp::core::messaging::SpikeData spikes;
    for (knp::core::messaging::SpikeIndex i = 0; i < population.size(); ++i)
    {
        bool was_reset = false;
        auto &neuron = population[i];
        if (neuron.potential_ >= neuron.activation_threshold_)
        {
            spikes.push_back(i);
            if (neuron.is_diff_) neuron.potential_ -= neuron.activation_threshold_;
            if (neuron.is_reset_)
            {
                neuron.potential_ = neuron.potential_reset_value_;
                was_reset = true;
            }
        }
        if (neuron.potential_ <= -static_cast<float>(neuron.negative_activation_threshold_) && !was_reset)
        {
            // Might probably want a negative spike, but we don't have any of the sort in Kaspersky Neuromorphic Platform. 
            // Not a large problem, just requires some conversion.
            if (neuron.saturate_)
            {
                neuron.potential_ = -static_cast<float>(neuron.negative_activation_threshold_);
                continue;
            }
            if (neuron.is_reset_)
                neuron.potential_ = -neuron.potential_reset_value_;
            else if (neuron.is_diff_)
                neuron.potential_ += neuron.negative_activation_threshold_;
        }
    }
    return spikes;
}

}  // namespace knp::backends::cpu
