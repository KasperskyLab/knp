/**
 * @file altai_lif.h
 * @brief AltAILIF neuron type traits.
 * @kaspersky_support Alexey Lubiakin
 * @date 02.05.2024
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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
#include <limits>
#include <numeric>
#include <utility>

#include "type_traits.h"


/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{

/**
 * @brief AltAILIF neuron.
 * 
 * @note Use as a template parameter only.
 */
struct AltAILIF;


/**
 * @brief Structure for AltAILIF neuron default values.
 */
template <>
struct default_values<AltAILIF>
{
    /**
     * @brief The parameter defines the default value for `is_diff_` flag of AltAILIF neuron.
     * 
     * @details If `is_diff_` flag is set to `true` and neuron potential exceeds one of its threshold value
     * after the neuron receives a spike, the `potential_` parameter takes a value by which
     * the potential threshold is exceeded.
     */
    constexpr static bool is_diff_ = false;

    /**
     * @brief The parameter defines the default value for `is_reset_` flag of AltAILIF neuron.
     * 
     * @details If `is_reset_` flag is set to `true` and neuron potential exceeds its threshold value
     * after the neuron receives a spike, the `potential_` parameter takes
     * a value of the `potential_reset_value_` parameter.
     */
    constexpr static bool is_reset_ = true;

    /**
     * @brief The parameter defines the default value for `leak_rev_` flag of AltAILIF neuron.
     * 
     * @details If `leak_rev_` flag is set to `true`, the `potential_leak_` sign automatically changes
     * along with the change of the `potential_` value sign.
     */
    constexpr static bool leak_rev_ = true;

    /**
     * @brief The parameter defines the default value for `saturate_` flag of AltAILIF neuron.
     * 
     * @details If `saturate_` flag is set to `true` and the neuron potential is less than
     * a negative `negative_activation_threshold_` value after the neuron receives a spike,
     * the `potential_` parameter takes the `negative_activation_threshold_` value.
     */
    constexpr static bool saturate_ = true;

    /**
     * @brief The parameter defines the default value for `do_not_save_` flag of AltAILIF neuron.
     * 
     * @details If `do_not_save_` flag is set to `false`, the `potential_` value is stored with each timestamp.
     */
    constexpr static bool do_not_save_ = false;

    /**
     * @brief The parameter defines the default value for `potential_` state of AltAILIF neuron.
     * 
     * @details Additional packet is sent to AltAI-1 for each neuron with non-zero initial `potential_` value.
     */
    constexpr static float potential_ = 0;

    /**
     * @brief The parameter defines the neuron potential value before synaptic impact.
     */
    constexpr static float pre_impact_potential_ = 0;

    /**
     * @brief The parameter defines the default value for `activation_threshold_` of AltAILIF neuron.
     */
    constexpr static uint16_t activation_threshold_ = 1;

    /**
     * @brief The parameter defines the default value for `negative_activation_threshold_` of AltAILIF neuron.
     * 
     * @details The default value was chosen for the scenario where negative spikes are not used and
     * `negative_activation_threshold_` with `saturate_` set to `true` are used to protect against `potential_` negative
     * overflow.
     */
    constexpr static uint16_t negative_activation_threshold_ = 30000;

    /**
     * @brief The parameter defines the default value for `potential_leak_` of AltAILIF neuron.
     */
    constexpr static int16_t potential_leak_ = 0;

    /**
     * @brief The parameter defines the default value for `potential_reset_value_` of AltAILIF neuron.
     */
    constexpr static uint16_t potential_reset_value_ = 0;

    /**
     * @brief The parameter defines a dopamine value that accumulates incoming dopamine‑synapse impacts.
     * 
     * @details The accumulated value can later be used by learning rules or other mechanisms that depend 
     * on the total dopamine received by the neuron during the step.
     */
    constexpr static double dopamine_value_ = 0.0;

    /**
     * @brief The parameter defines an additional part of the threshold for membrane potential.
     * 
     * @details The parameter is used for mechanisms that are implemented in specific neuron types.
     * 
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    constexpr static double additional_threshold_ = 0.;

    /**
     * @brief The parameter controls the activity timer for the neuron.
     * 
     * @details If this parameter is positive, the timer is decremented by 1 on each step. If it is negative, the timer 
     * is incremented by 1 on each step. If it equals 0, it is set to the largest positive value representable.
     * When the neuron receives a spike, this parameter is set to the weight of the synapse from which the spike originated.
     * The neuron will forward a spike only when this parameter is positive.
     */
    constexpr static int64_t activity_time_ = 0;
};


/**
 * @brief Structure for AltAILIF neuron parameters.
 */
template <>
struct neuron_parameters<AltAILIF>
{
    /**
     * @brief If `is_diff_` flag is set to `true` and neuron potential exceeds
     * one of its threshold value after the neuron receives a spike, the `potential_` parameter
     * takes a value by which the potential threshold is exceeded.
     * 
     * @details The code below demonstrates the logic of after-spike flags for AltAILIF neuron:
     * @code{.cpp}
     * if (potential_ >= activation_threshold_)
     * {
     *  if (is_diff_) potential_ -= activation_threshold_;
     *  if (is_reset_) potential_ = potential_reset_value_;
     * }
     * else if (potential_ < -negative_activation_threshold_)
     * {
     *  if (saturate_) potential_ = negative_activation_threshold_;
     *  else {
     *      if (is_reset_) potential_ = -potential_reset_value_;
     *      else if (is_diff_) potential_ -= negative_activation_threshold_;
     *  }
     * }
     * @endcode
     */
    bool is_diff_ = default_values<AltAILIF>::is_diff_;

    /**
     * @brief If `is_reset_` flag is set to `true` and neuron potential exceeds
     * its threshold value after the neuron receives a spike, the `potential_` parameter
     * takes a value of the `potential_reset_value_` parameter.
     * 
     * @details The code below demonstrates the logic of after-spike flags for AltAILIF neuron:
     * @code{.cpp}
     * if (potential_ >= activation_threshold_)
     * {
     *  if (is_diff_) potential_ -= activation_threshold_;
     *  if (is_reset_) potential_ = potential_reset_value_;
     * }
     * else if (potential_ < -negative_activation_threshold_)
     * {
     *  if (saturate_) potential_ = negative_activation_threshold_;
     *  else {
     *      if (is_reset_) potential_ = -potential_reset_value_;
     *      else if (is_diff_) potential_ -= negative_activation_threshold_;
     *  }
     * }
     * @endcode
     */
    bool is_reset_ = default_values<AltAILIF>::is_reset_;

    /**
     * @brief If `leak_rev_` flag is set to `true`, the `potential_leak_` sign automatically changes
     * along with the change of the `potential_` value sign.
     * 
     * @details The code below demonstrates the logic of `leak_rev` flag:
     * @code{.cpp}
     * if (leak_rev_)
     *  potential_ += (potential_ < 0 ? -potential_leak_ : potential_leak_);
     * else
     *  potential_ += potential_leak_;
     * @endcode
     */
    bool leak_rev_ = default_values<AltAILIF>::leak_rev_;

    /**
     * @brief If `saturate_` flag is set to `true` and the neuron potential is less than
     * a negative `negative_activation_threshold_` value after the neuron receives a spike,
     * the `potential_` parameter takes the `negative_activation_threshold_` value.
     * 
     * @details The code below demonstrates the logic of after-spike flags for AltAILIF neuron:
     * @code{.cpp}
     * if (potential_ >= activation_threshold_)
     * {
     *  if (is_diff_) potential_ -= activation_threshold_;
     *  if (is_reset_) potential_ = potential_reset_value_;
     * }
     * else if (potential_ < -negative_activation_threshold_)
     * {
     *  if (saturate_) potential_ = negative_activation_threshold_;
     *  else {
     *      if (is_reset_) potential_ = -potential_reset_value_;
     *      else if (is_diff_) potential_ -= negative_activation_threshold_;
     *  }
     * }
     * @endcode
     */
    bool saturate_ = default_values<AltAILIF>::saturate_;

    /**
     * @brief If `do_not_save_` flag is set to `false`, the `potential_` value is stored with each timestamp.
     * 
     * @details If set to `false`, the potential takes a value of the `potential_reset_value_` parameter
     * at the beginning of each subsequent time step (except the first time step, when neuron potential
     * takes the `potential_` default value).
     */
    bool do_not_save_ = default_values<AltAILIF>::do_not_save_;

    /**
     * @brief The parameter defines the neuron potential value.
     */
    float potential_ = default_values<AltAILIF>::potential_;

    /**
     * @brief The parameter defines the neuron potential value before synaptic impact.
     */
    float pre_impact_potential_ = default_values<AltAILIF>::pre_impact_potential_;

    /**
     * @brief The parameter defines the threshold value of neuron potential, after exceeding which a positive spike can
     * be emitted.
     * 
     * @details Positive spike is emitted if `potential_` >= `activation_threshold_`
     * and the neuron has a target for positive spike.
     */
    uint16_t activation_threshold_ = default_values<AltAILIF>::activation_threshold_;

    /**
     * @brief The parameter defines the threshold value of neuron potential, below which a negative spike can be
     * emitted.
     * 
     * @details Negative spike is emitted if `potential_` < -`negative_activation_threshold_`
     * and the neuron has a target for negative spike.
     */
    uint16_t negative_activation_threshold_ = default_values<AltAILIF>::negative_activation_threshold_;

    /**
     * @brief The parameter defines the constant leakage of the neuron potential.
     * 
     * @details The code below demonstrates how leakage mechanism works for AltAILIF neuron:
     * @code{.cpp}
     * if (leak_rev_)
     *  potential_ += (potential_ < 0 ? -potential_leak_ : potential_leak_);
     * else
     *  potential_ += potential_leak_;
     * @endcode
     */
    int16_t potential_leak_ = default_values<AltAILIF>::potential_leak_;

    /**
     * @brief The parameter defines a reset value of the neuron potential after one of the thresholds has been exceeded.
     * 
     * @details The code below demonstrates in what case `potential_reset_value_`
     * is used depending on all AltAILIF flags:
     * @code{.cpp}
     * if (potential_ >= activation_threshold_)
     * {
     *  if (is_diff_) potential_ -= activation_threshold_;
     *  if (is_reset_) potential_ = potential_reset_value_;
     * }
     * else if (potential_ < -negative_activation_threshold_)
     * {
     *  if (saturate_) potential_ = negative_activation_threshold_;
     *  else {
     *      if (is_reset_) potential_ = -potential_reset_value_;
     *      else if (is_diff_) potential_ -= negative_activation_threshold_;
     *  }
     * }
     * @endcode
     */
    uint16_t potential_reset_value_ = default_values<AltAILIF>::potential_reset_value_;

    /**
     * @brief The parameter defines a dopamine value that accumulates incoming dopamine‑synapse impacts.
     * 
     * @details The accumulated value can later be used by learning rules or other mechanisms that depend 
     * on the total dopamine received by the neuron during the step.
     */
    double dopamine_value_ = default_values<AltAILIF>::dopamine_value_;

    /**
     * @brief The parameter defines an additional part of the threshold for membrane potential.
     * 
     * @details The parameter is used for mechanisms that are implemented in specific neuron types.
     * 
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    double additional_threshold_ = default_values<AltAILIF>::additional_threshold_;

    /**
     * @brief The parameter controls the activity timer for the neuron.
     * 
     * @details If this parameter is positive, the timer is decremented by 1 on each step. If it is negative, the timer 
     * is incremented by 1 on each step. If it equals 0, it is set to the largest positive value representable.
     * When the neuron receives a spike, this parameter is set to the weight of the synapse from which the spike originated.
     * The neuron will forward a spike only when this parameter is positive.
     */
    int64_t activity_time_ = default_values<AltAILIF>::activity_time_;
};

}  // namespace knp::neuron_traits
