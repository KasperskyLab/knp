/**
 * @file impl.h
 * @kaspersky_support Postnikov D.
 * @date 08.12.2025
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

#include <limits>

#include "shared.h"

namespace knp::backends::cpu::populations::impl::altai
{

inline void calculate_pre_impact_single_neuron_state_impl(knp::neuron_traits::neuron_parameters<AltaiNeuron> &neuron)
{
    neuron.potential_ = std::round(neuron.potential_);

    neuron.potential_ = neuron.do_not_save_ ? static_cast<float>(neuron.potential_reset_value_) : neuron.potential_;

    neuron.pre_impact_potential_ = neuron.potential_;
}

inline void impact_neuron_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    switch (impact.synapse_type_)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            neuron.potential_ += impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            neuron.potential_ -= impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::DOPAMINE:
            neuron.dopamine_value_ += impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            if (((neuron.activity_time_ < 0 && impact.impact_value_ < 0) ||
                 (neuron.activity_time_ > 0 && impact.impact_value_ > 0)) &&
                std::abs(neuron.activity_time_) > std::abs(impact.impact_value_))
            {
            }
            else
                neuron.activity_time_ = static_cast<decltype(neuron.activity_time_)>(impact.impact_value_);
            break;
        default:
            break;
    }
}


inline bool calculate_post_impact_single_neuron_state_impl(knp::neuron_traits::neuron_parameters<AltaiNeuron> &neuron)
{
    {  // Potential leak.
        // -1 if leak_rev is true and potential < 0, 1 otherwise.
        const int sign = (neuron.leak_rev_ && neuron.potential_ < 0) ? -1 : 1;
        neuron.potential_ += neuron.potential_leak_ * sign;
    }

    bool spiked = false;
    if (neuron.activity_time_ > 0)
    {
        neuron.activity_time_--;
    }
    else if (neuron.activity_time_ < 0)
    {
        neuron.activity_time_++;
    }

    if (neuron.activity_time_ == 0)
    {
        neuron.activity_time_ = std::numeric_limits<decltype(neuron.activity_time_)>::max();
    }

    bool was_reset = false;
    if (neuron.potential_ >= neuron.activation_threshold_ + neuron.additional_threshold_)
    {
        if (neuron.activity_time_ > 0) spiked = true;
        if (neuron.is_diff_) neuron.potential_ -= neuron.activation_threshold_ + neuron.additional_threshold_;
        if (neuron.is_reset_)
        {
            neuron.potential_ = neuron.potential_reset_value_;
            was_reset = true;
        }
    }
    if (neuron.potential_ <= -static_cast<float>(neuron.negative_activation_threshold_) && !was_reset)
    {
        // Might probably want a negative spike, but we don't have any of the sort in KNP. Not a large problem,
        // just requires some conversion.
        if (neuron.saturate_)
        {
            neuron.potential_ = -static_cast<float>(neuron.negative_activation_threshold_);
        }
        else
        {
            if (neuron.is_reset_)
                neuron.potential_ = -neuron.potential_reset_value_;
            else if (neuron.is_diff_)
                neuron.potential_ += neuron.negative_activation_threshold_;
        }
    }

    return spiked;
}

}  //namespace knp::backends::cpu::populations::impl::altai
