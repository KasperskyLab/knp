/**
 * @file astrocyte_nonlinear_impl.h
 * @brief Spike accumulation and nonlinear calcium pulse dynamics.
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

#include <knp/astrocyte-traits/astrocyte_nonlinear.h>
#include <knp/core/messaging/synaptic_impact_message.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace knp::backends::cpu::populations::impl::astrocyte_nonlinear
{

/**
 * @brief Finish an expired pulse before accepting incoming spikes.
 * @param astrocyte astrocyte parameters.
 */
inline void calculate_pre_impact_single_neuron_state_impl(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteNonlinear> &astrocyte)
{
    if (astrocyte.pulse_active_ && astrocyte.pulse_time_ > astrocyte.pulse_duration_)
    {
        astrocyte.pulse_active_ = false;
        astrocyte.calcium_ = 0.F;
        astrocyte.pulse_time_ = 0.F;
    }
}


/**
 * @brief Count a spike only while the accumulation window is open.
 * @param astrocyte astrocyte parameters.
 * @param impact impact message; its weight is ignored.
 * @param is_forcing forcing flag; ignored.
 */
inline void impact_neuron_impl(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteNonlinear> &astrocyte,
    const knp::core::messaging::SynapticImpact & /* impact */, bool /* is_forcing */)
{
    if (!astrocyte.pulse_active_) ++astrocyte.monitored_spikes_this_step_;
}


/**
 * @brief Accumulate window spikes or calculate the current calcium sample.
 * @param astrocyte astrocyte parameters.
 * @return true if calcium reaches or exceeds the modulation threshold.
 */
inline bool calculate_post_impact_single_neuron_state_impl(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteNonlinear> &astrocyte)
{
    if (astrocyte.window_steps_ == 0 || astrocyte.window_stride_ == 0 || !(astrocyte.pulse_time_scale_ > 0.F) ||
        !std::isfinite(astrocyte.pulse_time_scale_) || !(astrocyte.dt_ > 0.F) || !std::isfinite(astrocyte.dt_) ||
        !(astrocyte.pulse_duration_ >= 0.F) || !std::isfinite(astrocyte.pulse_duration_))
        throw std::invalid_argument("Invalid nonlinear astrocyte time parameters.");

    const auto spikes = astrocyte.monitored_spikes_this_step_;
    astrocyte.monitored_spikes_this_step_ = 0;
    if (!astrocyte.pulse_active_)
    {
        astrocyte.calcium_ = 0.F;
        if (astrocyte.spike_history_.size() != astrocyte.window_steps_)
        {
            astrocyte.spike_history_.assign(astrocyte.window_steps_, 0);
            astrocyte.spike_counter_ = 0;
            astrocyte.window_step_ = 0;
            astrocyte.window_position_ = 0;
            astrocyte.window_check_step_ = 0;
        }
        astrocyte.spike_counter_ -= astrocyte.spike_history_.at(astrocyte.window_position_);
        astrocyte.spike_history_[astrocyte.window_position_] = spikes;
        astrocyte.spike_counter_ += spikes;
        astrocyte.window_position_ = (astrocyte.window_position_ + 1) % astrocyte.window_steps_;
        astrocyte.total_spikes_ += spikes;
        if (astrocyte.window_step_ < astrocyte.window_steps_)
        {
            if (++astrocyte.window_step_ < astrocyte.window_steps_) return false;
        }
        else
        {
            if (++astrocyte.window_check_step_ < astrocyte.window_stride_) return false;
            astrocyte.window_check_step_ = 0;
        }

        const auto count = std::min(astrocyte.spike_counter_, astrocyte.spike_count_max_);
        if (count == 0)
        {
            astrocyte.amplitude_ = 0.F;
            return false;
        }
        astrocyte.window_step_ = 0;
        astrocyte.window_position_ = 0;
        astrocyte.window_check_step_ = 0;
        astrocyte.spike_counter_ = 0;
        astrocyte.spike_history_.clear();
        astrocyte.amplitude_ =
            astrocyte.amplitude_max_ / (1.0 + std::exp(
                                                  -static_cast<double>(astrocyte.amplitude_slope_) *
                                                  (static_cast<double>(count) - astrocyte.spike_midpoint_)));
        astrocyte.pulse_active_ = true;
        astrocyte.pulse_time_ = 0.F;
        ++astrocyte.activation_count_;
    }

    const double time = static_cast<double>(astrocyte.pulse_time_) / astrocyte.pulse_time_scale_;
    const double rise = 1.0 / (1.0 + std::exp(-astrocyte.rise_slope_ * (time - astrocyte.rise_time_)));
    const double fall_time = astrocyte.fall_time_scale_ * astrocyte.amplitude_;
    const double fall = 1.0 / (1.0 + std::exp(-astrocyte.fall_slope_ * (time - fall_time)));
    astrocyte.calcium_ = astrocyte.amplitude_ * rise * (1.0 - fall);
    astrocyte.pulse_time_ += astrocyte.dt_;
    return astrocyte.calcium_ >= astrocyte.modulation_threshold_;
}

}  // namespace knp::backends::cpu::populations::impl::astrocyte_nonlinear
