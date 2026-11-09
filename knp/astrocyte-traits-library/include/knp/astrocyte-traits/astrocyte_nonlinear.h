/**
 * @file astrocyte_nonlinear.h
 * @brief Nonlinear astrocyte type traits and parameters.
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

#include <cstdint>
#include <vector>

#include "type_traits.h"


namespace knp::astrocyte_traits
{

/**
 * @brief Astrocyte model with a sigmoid amplitude response and a calcium pulse formed by two sigmoids.
 */
struct AstrocyteNonlinear;


/**
 * @brief Default values for the nonlinear astrocyte model.
 */
template <>
struct default_values<AstrocyteNonlinear>
{
    /**
     * @brief Default value of `calcium_`.
     */
    constexpr static float calcium_ = 0.F;

    /**
     * @brief Default value of `window_stride_`.
     */
    constexpr static uint64_t window_stride_ = 5;

    /**
     * @brief Default value of `window_position_`.
     */
    constexpr static uint64_t window_position_ = 0;

    /**
     * @brief Default value of `window_check_step_`.
     */
    constexpr static uint64_t window_check_step_ = 0;

    /**
     * @brief Default value of `spike_count_max_`.
     */
    constexpr static uint64_t spike_count_max_ = 20;

    /**
     * @brief Default value of `pulse_time_scale_`.
     */
    constexpr static float pulse_time_scale_ = 100.F;

    /**
     * @brief Default value of `window_step_`.
     */
    constexpr static uint64_t window_step_ = 0;

    /**
     * @brief Default value of `spike_counter_`.
     */
    constexpr static uint64_t spike_counter_ = 0;

    /**
     * @brief Default value of `total_spikes_`.
     */
    constexpr static uint64_t total_spikes_ = 0;

    /**
     * @brief Default value of `activation_count_`.
     */
    constexpr static uint64_t activation_count_ = 0;

    /**
     * @brief Default value of `monitored_spikes_this_step_`.
     */
    constexpr static uint64_t monitored_spikes_this_step_ = 0;

    /**
     * @brief Default value of `pulse_active_`.
     */
    constexpr static bool pulse_active_ = false;

    /**
     * @brief Default value of `pulse_time_`.
     */
    constexpr static float pulse_time_ = 0.F;

    /**
     * @brief Default value of `amplitude_`.
     */
    constexpr static float amplitude_ = 0.F;

    /**
     * @brief Default value of `window_steps_`.
     */
    constexpr static uint64_t window_steps_ = 50;

    /**
     * @brief Default value of `dt_`.
     */
    constexpr static float dt_ = 1.F;

    /**
     * @brief Default value of `pulse_duration_`.
     */
    constexpr static float pulse_duration_ = 1000.F;

    /**
     * @brief Default value of `amplitude_max_`.
     */
    constexpr static float amplitude_max_ = 0.8F;

    /**
     * @brief Default value of `amplitude_slope_`.
     */
    constexpr static float amplitude_slope_ = 0.5F;

    /**
     * @brief Default value of `spike_midpoint_`.
     */
    constexpr static float spike_midpoint_ = 8.F;

    /**
     * @brief Default value of `rise_slope_`.
     */
    constexpr static float rise_slope_ = 5.F;

    /**
     * @brief Default value of `fall_slope_`.
     */
    constexpr static float fall_slope_ = 2.F;

    /**
     * @brief Default value of `rise_time_`.
     */
    constexpr static float rise_time_ = 1.F;

    /**
     * @brief Default value of `fall_time_scale_`.
     */
    constexpr static float fall_time_scale_ = 8.F;

    /**
     * @brief Default value of `modulation_threshold_`.
     */
    constexpr static float modulation_threshold_ = 0.2F;

    /**
     * @brief Default value of `active_weight_multiplier_`.
     */
    constexpr static float active_weight_multiplier_ = 1000.F;

    /**
     * @brief Default value of `min_weight_`.
     */
    constexpr static float min_weight_ = 0.001F;
};


/**
 * @brief Parameters of the nonlinear astrocyte model.
 *
 * @details Spikes are counted in a sliding window checked at regular intervals. A nonempty window starts one pulse;
 * incoming spikes are ignored until the pulse ends. Empty windows do not produce calcium pulses.
 */
template <>
struct astrocyte_parameters<AstrocyteNonlinear>
{
    /**
     * @brief Current calcium level.
     */
    float calcium_ = default_values<AstrocyteNonlinear>::calcium_;

    /**
     * @brief Number of steps between checks of a full spike window.
     */
    uint64_t window_stride_ = default_values<AstrocyteNonlinear>::window_stride_;

    /**
     * @brief Position of the next entry in the spike history.
     */
    uint64_t window_position_ = default_values<AstrocyteNonlinear>::window_position_;

    /**
     * @brief Steps elapsed since the last full-window check.
     */
    uint64_t window_check_step_ = default_values<AstrocyteNonlinear>::window_check_step_;

    /**
     * @brief Upper limit of the spike count used to calculate pulse amplitude.
     */
    uint64_t spike_count_max_ = default_values<AstrocyteNonlinear>::spike_count_max_;

    /**
     * @brief Number of network ticks per time unit of the calcium formula.
     */
    float pulse_time_scale_ = default_values<AstrocyteNonlinear>::pulse_time_scale_;

    /**
     * @brief Circular history of monitored spike counts, one entry per window step.
     */
    std::vector<uint64_t> spike_history_;

    /**
     * @brief Number of filled steps in the current sliding window.
     */
    uint64_t window_step_ = default_values<AstrocyteNonlinear>::window_step_;

    /**
     * @brief Number of spikes accumulated in the current window.
     */
    uint64_t spike_counter_ = default_values<AstrocyteNonlinear>::spike_counter_;

    /**
     * @brief Total number of spikes accepted during accumulation windows.
     */
    uint64_t total_spikes_ = default_values<AstrocyteNonlinear>::total_spikes_;

    /**
     * @brief Number of calcium pulses started.
     */
    uint64_t activation_count_ = default_values<AstrocyteNonlinear>::activation_count_;

    /**
     * @brief Number of accepted incoming spikes during this step.
     */
    uint64_t monitored_spikes_this_step_ = default_values<AstrocyteNonlinear>::monitored_spikes_this_step_;

    /**
     * @brief True while a calcium pulse is being evaluated.
     */
    bool pulse_active_ = default_values<AstrocyteNonlinear>::pulse_active_;

    /**
     * @brief Time of the next calcium sample within the pulse.
     */
    float pulse_time_ = default_values<AstrocyteNonlinear>::pulse_time_;

    /**
     * @brief Amplitude determined by the preceding accumulation window.
     */
    float amplitude_ = default_values<AstrocyteNonlinear>::amplitude_;

    /**
     * @brief Number of steps in a spike accumulation window.
     */
    uint64_t window_steps_ = default_values<AstrocyteNonlinear>::window_steps_;

    /**
     * @brief Time increment per simulation step, in KNP ticks.
     */
    float dt_ = default_values<AstrocyteNonlinear>::dt_;

    /**
     * @brief Last time point of the calcium pulse.
     */
    float pulse_duration_ = default_values<AstrocyteNonlinear>::pulse_duration_;

    /**
     * @brief Upper limit of the calcium amplitude.
     */
    float amplitude_max_ = default_values<AstrocyteNonlinear>::amplitude_max_;

    /**
     * @brief Slope of the amplitude response to spike count.
     */
    float amplitude_slope_ = default_values<AstrocyteNonlinear>::amplitude_slope_;

    /**
     * @brief Spike count at half of the maximum amplitude.
     */
    float spike_midpoint_ = default_values<AstrocyteNonlinear>::spike_midpoint_;

    /**
     * @brief Slope of the rising sigmoid.
     */
    float rise_slope_ = default_values<AstrocyteNonlinear>::rise_slope_;

    /**
     * @brief Slope of the falling sigmoid.
     */
    float fall_slope_ = default_values<AstrocyteNonlinear>::fall_slope_;

    /**
     * @brief Midpoint of the rising sigmoid.
     */
    float rise_time_ = default_values<AstrocyteNonlinear>::rise_time_;

    /**
     * @brief Factor relating calcium amplitude to the fall midpoint.
     */
    float fall_time_scale_ = default_values<AstrocyteNonlinear>::fall_time_scale_;

    /**
     * @brief Minimum calcium level at which synapse weights are modulated.
     */
    float modulation_threshold_ = default_values<AstrocyteNonlinear>::modulation_threshold_;

    /**
     * @brief Weight multiplier while calcium reaches or exceeds the threshold.
     */
    float active_weight_multiplier_ = default_values<AstrocyteNonlinear>::active_weight_multiplier_;

    /**
     * @brief Lower bound for the modulated synapse weight.
     */
    float min_weight_ = default_values<AstrocyteNonlinear>::min_weight_;
};

}  // namespace knp::astrocyte_traits
