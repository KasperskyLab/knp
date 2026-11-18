/**
 * @file astrocyte_simple_impl.h
 * @brief Calcium dynamics and spike accumulation for astrocytes.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <knp/astrocyte-traits/astrocyte_simple.h>
#include <knp/core/population.h>

#include <stdexcept>


namespace knp::backends::cpu::populations::impl::astrocyte_simple
{

/**
 * @brief No state update is required before synaptic impacts.
 */
inline void calculate_pre_impact_single_neuron_state_impl(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteSimple> & /* astrocyte */)
{
}


/**
 * @brief Count one monitored spike for each incoming synaptic impact.
 * @param astrocyte astrocyte parameters.
 * @note Impact weight and forcing flag do not affect the spike count.
 */
inline void impact_neuron_impl(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteSimple> &astrocyte,
    const knp::core::messaging::SynapticImpact & /* impact */, bool /* is_forcing */)
{
    ++astrocyte.monitored_spikes_this_step_;
}


/**
 * @brief Update spike counters and advance the calcium rise or decay by one step.
 * @param astrocyte astrocyte parameters.
 * @return true if the resulting calcium level reaches or exceeds the modulation threshold.
 * @note Incoming spikes contribute to the counters only on steps that start with zero calcium.
 * The per-step spike count is reset after reading.
 */
inline bool calculate_post_impact_single_neuron_state_impl(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteSimple> &astrocyte)
{
    if (astrocyte.window_steps_ == 0 || astrocyte.window_stride_ == 0)
        throw std::invalid_argument("Simple astrocyte window width and stride must be positive.");

    const uint64_t spikes_this_step = astrocyte.monitored_spikes_this_step_;
    astrocyte.monitored_spikes_this_step_ = 0;

    if (astrocyte.calcium_ == 0.F)
    {
        if (astrocyte.spike_history_.size() != astrocyte.window_steps_)
        {
            astrocyte.spike_history_.assign(astrocyte.window_steps_, 0);
            astrocyte.spike_counter_ = 0;
            astrocyte.window_step_ = 0;
            astrocyte.window_position_ = 0;
            astrocyte.window_check_step_ = 0;
        }
        astrocyte.spike_counter_ -= astrocyte.spike_history_.at(astrocyte.window_position_);
        astrocyte.spike_history_[astrocyte.window_position_] = spikes_this_step;
        astrocyte.spike_counter_ += spikes_this_step;
        astrocyte.window_position_ = (astrocyte.window_position_ + 1) % astrocyte.window_steps_;
        astrocyte.total_spikes_ += spikes_this_step;

        bool check_window = false;
        if (astrocyte.window_step_ < astrocyte.window_steps_)
        {
            check_window = ++astrocyte.window_step_ == astrocyte.window_steps_;
        }
        else if (++astrocyte.window_check_step_ >= astrocyte.window_stride_)
        {
            astrocyte.window_check_step_ = 0;
            check_window = true;
        }

        if (check_window && astrocyte.spike_counter_ >= astrocyte.spike_threshold_)
        {
            astrocyte.rising_ = true;
            astrocyte.calcium_ = astrocyte.calcium_step_;
            astrocyte.spike_counter_ = 0;
            astrocyte.spike_history_.clear();
            astrocyte.window_step_ = 0;
            astrocyte.window_position_ = 0;
            astrocyte.window_check_step_ = 0;
            ++astrocyte.activation_count_;
        }
    }
    else if (astrocyte.rising_)
    {
        astrocyte.calcium_ += astrocyte.calcium_step_;
        if (astrocyte.calcium_ >= astrocyte.calcium_max_)
        {
            astrocyte.calcium_ = astrocyte.calcium_max_;
            astrocyte.rising_ = false;
        }
    }
    else
    {
        astrocyte.calcium_ -= astrocyte.calcium_step_;
        if (astrocyte.calcium_ < 0.F)
        {
            astrocyte.calcium_ = 0.F;
            astrocyte.rising_ = false;
        }
    }

    return astrocyte.calcium_ >= astrocyte.modulation_threshold_;
}

}  // namespace knp::backends::cpu::populations::impl::astrocyte_simple
