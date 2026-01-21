/**
 * @file impl.h
 * @kaspersky_support Postnikov D.
 * @date 12.12.2025
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

#include "stdp.h"


namespace knp::backends::cpu::populations::impl::blifat
{

inline void calculate_pre_impact_single_neuron_state_impl(knp::neuron_traits::neuron_parameters<BlifatNeuron> &neuron)
{
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
    {
        neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
    }
    else
    {
        neuron.potential_ *= neuron.potential_decay_;
    }
    neuron.pre_impact_potential_ = neuron.potential_;
}


inline void calculate_pre_impact_single_neuron_state_impl(
    knp::neuron_traits::neuron_parameters<STDPBlifatNeuron> &neuron)
{
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    neuron.dopamine_value_ = 0.0;
    neuron.is_being_forced_ = false;

    if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
    {
        neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
    }
    else
    {
        neuron.potential_ *= neuron.potential_decay_;
    }
    neuron.pre_impact_potential_ = neuron.potential_;
}


inline void impact_neuron_impl(
    knp::neuron_traits::neuron_parameters<BlifatNeuron> &neuron, const knp::core::messaging::SynapticImpact &impact,
    bool is_forcing)
{
    switch (impact.synapse_type_)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            neuron.potential_ += impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            neuron.potential_ -= impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
            neuron.inhibitory_conductance_ += impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::DOPAMINE:
            neuron.dopamine_value_ += impact.impact_value_;
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            neuron.total_blocking_period_ = static_cast<decltype(neuron.total_blocking_period_)>(impact.impact_value_);
            break;
        default:
            throw std::runtime_error("Unhandled synapse type.");
    }
}


inline void impact_neuron_impl(
    knp::neuron_traits::neuron_parameters<STDPBlifatNeuron> &neuron, const knp::core::messaging::SynapticImpact &impact,
    bool is_forcing)
{
    impact_neuron_impl(static_cast<knp::neuron_traits::neuron_parameters<BlifatNeuron> &>(neuron), impact, is_forcing);
    if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
    {
        neuron.is_being_forced_ |= is_forcing;
    }
}


inline bool calculate_post_impact_single_neuron_state_impl(knp::neuron_traits::neuron_parameters<BlifatNeuron> &neuron)
{
    bool spike = false;
    if (neuron.total_blocking_period_ <= 0)
    {
        // TODO: Make it more readable, don't be afraid to use if operators.
        // Restore potential that the neuron had before impacts.
        neuron.potential_ = neuron.pre_impact_potential_;
        bool was_negative = neuron.total_blocking_period_ < 0;
        // If it is negative, increase by 1.
        neuron.total_blocking_period_ += was_negative;
        if ((neuron.total_blocking_period_ == 0) && was_negative)
        {
            neuron.total_blocking_period_ = std::numeric_limits<int64_t>::max();
        }
    }
    else
    {
        neuron.total_blocking_period_ -= 1;
    }

    if (neuron.inhibitory_conductance_ < 1.0)
    {
        neuron.potential_ -=
            (neuron.potential_ - neuron.reversal_inhibitory_potential_) * neuron.inhibitory_conductance_;
    }
    else
    {
        neuron.potential_ = neuron.reversal_inhibitory_potential_;
    }

    // Three components of neuron threshold: "static", "common dynamic" and "implementation-specific dynamic".
    if ((neuron.n_time_steps_since_last_firing_ > neuron.absolute_refractory_period_) &&
        (neuron.potential_ >= neuron.activation_threshold_ + neuron.dynamic_threshold_ + neuron.additional_threshold_))
    {
        // Spike.
        neuron.dynamic_threshold_ += neuron.threshold_increment_;
        neuron.postsynaptic_trace_ += neuron.postsynaptic_trace_increment_;

        neuron.potential_ = neuron.potential_reset_value_;
        neuron.bursting_phase_ = neuron.bursting_period_;
        neuron.n_time_steps_since_last_firing_ = 0;
        spike = true;
    }

    if (neuron.potential_ < neuron.min_potential_)
    {
        neuron.potential_ = neuron.min_potential_;
    }

    return spike;
}
}  //namespace knp::backends::cpu::populations::impl::blifat
