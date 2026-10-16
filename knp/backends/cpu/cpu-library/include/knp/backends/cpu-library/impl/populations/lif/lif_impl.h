/**
 * @file lif_impl.h
 * @brief Implementation of lif neuron population.
 * @kaspersky_support Postnikov D.
 * @date 12.05.2026
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

#include <knp/core/population.h>

#include <spdlog/spdlog.h>

#include <limits>


namespace knp::backends::cpu::populations::impl::lif
{

inline void calculate_pre_impact_single_neuron_state_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron> &neuron)
{
    if (0 == neuron.refract_counter_)
    {
        neuron.potential_ *= neuron.leak_coefficient_;
    }
}


inline void impact_neuron_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    switch (impact.synapse_type_)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            if (0 == neuron.refract_counter_)
            {
                neuron.potential_ += impact.impact_value_;
            }
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            if (0 == neuron.refract_counter_)
            {
                neuron.potential_ -= impact.impact_value_;
            }
            break;

        default:
            SPDLOG_ERROR("Unhandled synapse type.");
            throw std::runtime_error("Unhandled synapse type.");
    }
}


inline bool calculate_post_impact_single_neuron_state_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::LIFNeuron> &neuron)
{
    if (0 == neuron.refract_counter_)
    {
        if (neuron.potential_ > neuron.activation_threshold_)
        {
            neuron.potential_ = neuron.potential_reset_value_;
            neuron.refract_counter_ = neuron.refract_period_;
            return true;
        }
    }
    else
    {
        --neuron.refract_counter_;
    }
    return false;
}

}  //namespace knp::backends::cpu::populations::impl::lif
