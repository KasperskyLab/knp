/**
 * @file astrocyte_simple.h
 * @brief Calcium-based astrocyte type traits and parameters.
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


/**
 * @brief Namespace for astrocyte traits.
 */
namespace knp::astrocyte_traits
{

/**
 * @brief Type that represents a calcium-based astrocyte.
 *
 * @note This type is intended to be used only as a template argument. It does not contain any data members.
 */
struct AstrocyteSimple;


/**
 * @brief Structure for astrocyte default values.
 */
template <>
struct default_values<AstrocyteSimple>
{
    /**
     * @brief Default value of `calcium_`.
     */
    constexpr static float calcium_ = 0.F;

    /**
     * @brief Default value of `window_steps_`.
     */
    constexpr static uint64_t window_steps_ = 50;

    /**
     * @brief Default value of `window_stride_`.
     */
    constexpr static uint64_t window_stride_ = 5;

    /**
     * @brief Default value of `window_step_`.
     */
    constexpr static uint64_t window_step_ = 0;

    /**
     * @brief Default value of `window_position_`.
     */
    constexpr static uint64_t window_position_ = 0;

    /**
     * @brief Default value of `window_check_step_`.
     */
    constexpr static uint64_t window_check_step_ = 0;

    /**
     * @brief Default value of `rising_`.
     */
    constexpr static bool rising_ = false;

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
     * @brief Default value of `calcium_max_`.
     */
    constexpr static float calcium_max_ = 1.F;

    /**
     * @brief Default value of `calcium_step_`.
     */
    constexpr static float calcium_step_ = 0.002F;

    /**
     * @brief Default value of `modulation_threshold_`.
     */
    constexpr static float modulation_threshold_ = 0.2F;

    /**
     * @brief Default value of `spike_threshold_`.
     */
    constexpr static uint64_t spike_threshold_ = 14;

    /**
     * @brief Default value of `base_weight_`.
     */
    constexpr static float base_weight_ = 0.07F;

    /**
     * @brief Default value of `active_weight_multiplier_`.
     */
    constexpr static float active_weight_multiplier_ = 1000.F;

    /**
     * @brief Default value of `min_weight_`.
     */
    constexpr static float min_weight_ = 0.001F;

    /**
     * @brief Default value of `max_weight_`.
     */
    constexpr static float max_weight_ = 2.F;
};


/**
 * @brief Structure for astrocyte parameters.
 */
template <>
struct astrocyte_parameters<AstrocyteSimple>
{
    /**
     * @brief Current calcium level.
     */
    float calcium_ = default_values<AstrocyteSimple>::calcium_;

    /**
     * @brief Number of steps in the sliding spike window.
     */
    uint64_t window_steps_ = default_values<AstrocyteSimple>::window_steps_;

    /**
     * @brief Number of steps between checks of a full spike window.
     */
    uint64_t window_stride_ = default_values<AstrocyteSimple>::window_stride_;

    /**
     * @brief Number of filled steps in the current spike window.
     */
    uint64_t window_step_ = default_values<AstrocyteSimple>::window_step_;

    /**
     * @brief Position of the next entry in the spike history.
     */
    uint64_t window_position_ = default_values<AstrocyteSimple>::window_position_;

    /**
     * @brief Steps elapsed since the last full-window check.
     */
    uint64_t window_check_step_ = default_values<AstrocyteSimple>::window_check_step_;

    /**
     * @brief Circular history of monitored spike counts, one entry per window step.
     */
    std::vector<uint64_t> spike_history_;

    /**
     * @brief `true` while calcium is rising towards `calcium_max_`, `false` while it decays back towards 0.
     */
    bool rising_ = default_values<AstrocyteSimple>::rising_;

    /**
     * @brief Number of monitored spikes in the current sliding window.
     * @note Updated only at zero calcium; reset when a calcium rise is activated.
     */
    uint64_t spike_counter_ = default_values<AstrocyteSimple>::spike_counter_;

    /**
     * @brief Cumulative number of monitored spikes counted on steps that start with zero calcium.
     * @note Spikes received while calcium is nonzero do not contribute to this counter.
     */
    uint64_t total_spikes_ = default_values<AstrocyteSimple>::total_spikes_;

    /**
     * @brief Number of times the calcium rise has been activated.
     */
    uint64_t activation_count_ = default_values<AstrocyteSimple>::activation_count_;

    /**
     * @brief Number of monitored spikes received during the current step.
     */
    uint64_t monitored_spikes_this_step_ = default_values<AstrocyteSimple>::monitored_spikes_this_step_;

    /**
     * @brief Calcium level at which calcium stops rising and starts decaying.
     */
    float calcium_max_ = default_values<AstrocyteSimple>::calcium_max_;

    /**
     * @brief Amount by which calcium rises or decays on every active step.
     */
    float calcium_step_ = default_values<AstrocyteSimple>::calcium_step_;

    /**
     * @brief Minimum calcium level at which the astrocyte modulates synapse weights.
     */
    float modulation_threshold_ = default_values<AstrocyteSimple>::modulation_threshold_;

    /**
     * @brief Number of monitored spikes required to activate the calcium rise.
     */
    uint64_t spike_threshold_ = default_values<AstrocyteSimple>::spike_threshold_;

    /**
     * @brief Base synapse weight; not used by the current multiplicative modulation.
     */
    float base_weight_ = default_values<AstrocyteSimple>::base_weight_;

    /**
     * @brief Multiplier applied to a controlled synapse's own weight while the astrocyte is active.
     */
    float active_weight_multiplier_ = default_values<AstrocyteSimple>::active_weight_multiplier_;

    /**
     * @brief Lower clamp for the resulting synapse weight.
     */
    float min_weight_ = default_values<AstrocyteSimple>::min_weight_;

    /**
     * @brief Maximum synapse weight; not applied by the current modulation.
     */
    float max_weight_ = default_values<AstrocyteSimple>::max_weight_;
};

}  // namespace knp::astrocyte_traits
