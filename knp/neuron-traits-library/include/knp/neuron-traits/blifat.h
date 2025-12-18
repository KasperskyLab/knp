/**
 * @file blifat.h
 * @brief BLIFAT neuron type traits.
 * @kaspersky_support Artiom N.
 * @date 26.01.2023
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
 * @brief BLIFAT neuron. Use as a template parameter only.
 */
struct BLIFATNeuron;


/**
 * @brief Structure for BLIFAT neuron default values.
 */
template <>
struct default_values<BLIFATNeuron>
{
    /**
     * @brief The parameter defines the default value of `n_time_steps_since_last_firing_` for a BLIFAT neuron.
     */
    constexpr static std::size_t n_time_steps_since_last_firing_ = std::numeric_limits<std::size_t>::max();

    /**
     * @brief The parameter defines a value to which membrane potential tends (for conductance-based inhibitory
     * synapses)
     */
    constexpr static double reversal_inhibitory_potential_ = -0.3;

    /**
     * @brief The parameter defines a value to which membrane potential tends (for current-based inhibitory synapses).
     */
    constexpr static double min_potential_ = -1.0e9;

    /**
     * @brief The parameter defines a constant part of the threshold for membrane potential.
     * @details After membrane potential reaches the threshold value, the neuron generates a spike.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_. 
     */
    constexpr static double activation_threshold_ = 1.0;

    /**
     * @brief The parameter defines a dynamic part of the threshold for membrane potential.
     * @details After neuron generates a spike, the `dynamic_threshold_` value increases by 
     * the `threshold_increment_` value.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    constexpr static double dynamic_threshold_ = 0.;

    /**
     * @brief The parameter defines an additional part of the threshold for membrane potential.
     * @details The parameter is used for mechanisms that are implemented in specific neuron types.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    constexpr static double additional_threshold_ = 0.;

    /**
     * @brief The parameter defines a time constant during which the `dynamic_threshold_` parameter tends to its base
     * value if nothing happens.
     */
    constexpr static double threshold_decay_ = 0.;

    /**
     * @brief The parameter defines a value that increases the `dynamic_threshold_` value if a neuron generates a spike.
     */
    constexpr static double threshold_increment_ = 0.;

    /**
     * @brief The parameter defines a threshold after reaching which a neuron generates spikes.
     */
    constexpr static double postsynaptic_trace_ = 0.;

    /**
     * @brief The parameter defines a time constant during which the `postsynaptic_trace_` parameter tends to zero if
     * nothing happens.
     * @details If `postsynaptic_trace_decay_` equals `0`, then `postsynaptic_trace_` also equals `0`.
     */
    constexpr static double postsynaptic_trace_decay_ = 0.;

    /**
     * @brief The parameter defines a value that increases the `postsynaptic_trace_` value if a neuron generates a
     * spike.
     */
    constexpr static double postsynaptic_trace_increment_ = 0.;

    /**
     * @brief The parameter defines speed with which a potential tends to the `reversal_inhibitory_potential` value.
     */
    constexpr static double inhibitory_conductance_ = 0.;

    /**
     * @brief The parameter defines a time constant during which the `inhibitory_conductance_` value decreases.
     */
    constexpr static double inhibitory_conductance_decay_ = 0.;

    /**
     * @brief The parameter defines the current membrane potential.
     */
    constexpr static double potential_ = 0;

    /**
     * @brief This parameter is used if there was a blocking signal.
     * @details If used, all potential changes due to synapses are ignored.
     */
    constexpr static double pre_impact_potential_ = 0;

    /**
     * @brief The parameter defines a time constant during which the `potential_` value tends to zero.
     */
    constexpr static double potential_decay_ = 0;

    /**
     * @brief The parameter defines a counter for the `bursting_period_` value.
     */
    constexpr static unsigned bursting_phase_ = 0;

    /**
     * @brief The parameter defines a number of network steps after reaching which a neuron generates a spike.
     * @details Value of 0 means that no bursting occurs.
     */
    constexpr static unsigned bursting_period_ = 0;

    /**
     * @brief The parameter defines a value that increases the membrane potential after a neuron generates a spike.
     */
    constexpr static double reflexive_weight_ = 0;

    /**
     * @brief The parameter defines a minimum number of network steps before a neuron can generate the next spike.
     */
    constexpr static unsigned absolute_refractory_period_ = 0;

    /**
     * @brief The parameter defines a potential value after a neuron generates a spike.
     */
    constexpr static double potential_reset_value_ = 0.;

    /**
     * @brief The parameter defines the default value for the number of network execution steps,
     * during which the neuron activity is totally blocked.
     */
    constexpr static int64_t total_blocking_period_ = std::numeric_limits<int64_t>::max();

    /**
     * @brief The parameter defines a dopamine value used to sum up all incoming dopamine synapse impacts.
     */
    constexpr static double dopamine_value_ = 0.0;

    /**
     * @brief The parameter defines stochastic stimulation - random number added to the potential every tick.
     */
    constexpr static double stochastic_stimulation_ = 0.0;

    /**
     * @brief The random number generator is used for stochastic stimulation.
     */
    constexpr static unsigned random_number_generator_state_ = 0;
};


/**
 * @brief Structure for BLIFAT neuron parameters.
 */
template <>
struct neuron_parameters<BLIFATNeuron>
{
    /**
     * @brief The parameter defines a number of network steps since the last spike.
     */
    std::size_t n_time_steps_since_last_firing_ = default_values<BLIFATNeuron>::n_time_steps_since_last_firing_;

    /**
     * @brief The parameter defines a constant part of the threshold for membrane potential.
     * @details After membrane potential reaches the threshold value, the neuron generates a spike.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_. 
     */
    double activation_threshold_ = default_values<BLIFATNeuron>::activation_threshold_;
    /**
     * @brief The parameter defines an additional part of the threshold for membrane potential.
     * @details The parameter is used for mechanisms that are implemented in specific neuron types.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    double additional_threshold_ = default_values<BLIFATNeuron>::additional_threshold_;
    /**
     * @brief The parameter defines a dynamic part of the threshold for membrane potential.
     * @details After neuron generates a spike, the `dynamic_threshold_` value increases by 
     * the `threshold_increment_` value.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    double dynamic_threshold_ = default_values<BLIFATNeuron>::dynamic_threshold_;
    /**
     * @brief The parameter defines a time constant during which the `dynamic_threshold_` parameter tends to its base
     * value if nothing happens.
     */
    double threshold_decay_ = default_values<BLIFATNeuron>::threshold_decay_;
    /**
     * @brief The parameter defines a value that increases the `dynamic_threshold_` value if a neuron generates a spike.
     */
    double threshold_increment_ = default_values<BLIFATNeuron>::threshold_increment_;
    /**
     * @brief The parameter defines a threshold after reaching which a neuron generates spikes.
     */
    double postsynaptic_trace_ = default_values<BLIFATNeuron>::postsynaptic_trace_;
    /**
     * @brief The parameter defines a time constant during which the `postsynaptic_trace_` parameter tends to zero if
     * nothing happens.
     * @details If `postsynaptic_trace_decay_` equals `0`, then `postsynaptic_trace_` also equals `0`.
     */
    double postsynaptic_trace_decay_ = default_values<BLIFATNeuron>::postsynaptic_trace_decay_;
    /**
     * @brief The parameter defines a value that increases the `postsynaptic_trace_` value if a neuron generates a
     * spike.
     */
    double postsynaptic_trace_increment_ = default_values<BLIFATNeuron>::postsynaptic_trace_increment_;
    /**
     * @brief The parameter defines speed with which a potential tends to the `reversal_inhibitory_potential` value.
     */
    double inhibitory_conductance_ = default_values<BLIFATNeuron>::inhibitory_conductance_;

    /**
     * @brief The parameter defines a time constant during which the `inhibitory_conductance_` value decreases.
     */
    double inhibitory_conductance_decay_ = default_values<BLIFATNeuron>::inhibitory_conductance_decay_;
    /**
     * @brief The parameter defines the current membrane potential.
     */
    double potential_ = default_values<BLIFATNeuron>::potential_;
    /**
     * @brief This parameter is used if there was a blocking signal.
     * @details If used, all potential changes due to synapses are ignored.
     */
    double pre_impact_potential_ = default_values<BLIFATNeuron>::pre_impact_potential_;
    /**
     * @brief The parameter defines a time constant during which the `potential_` value tends to zero.
     */
    double potential_decay_ = default_values<BLIFATNeuron>::potential_decay_;

    /**
     * @brief The parameter defines a counter for the `bursting_period_` value.
     */
    unsigned bursting_phase_ = default_values<BLIFATNeuron>::bursting_phase_;

    /**
     * @brief The parameter defines a number of network steps after reaching which a neuron generates a spike.
     * @details Value of 0 means that no bursting occurs.
     */
    unsigned bursting_period_ = default_values<BLIFATNeuron>::bursting_period_;
    /**
     * @brief The parameter defines a value that increases the membrane potential after a neuron generates a spike.
     */
    double reflexive_weight_ = default_values<BLIFATNeuron>::reflexive_weight_;

    /**
     * @brief The parameter takes the default value of `reversal_inhibitory_potential` defined for a BLIFAT neuron.
     */
    double reversal_inhibitory_potential_ = default_values<BLIFATNeuron>::reversal_inhibitory_potential_;

    /**
     * @brief The parameter defines a minimum number of network steps before a neuron can generate the next spike.
     */
    unsigned absolute_refractory_period_ = default_values<BLIFATNeuron>::absolute_refractory_period_;
    /**
     * @brief The parameter defines a potential value after a neuron generates a spike.
     */
    double potential_reset_value_ = default_values<BLIFATNeuron>::potential_reset_value_;

    /**
     * @brief The parameter takes the default value of `min_potential` defined for a BLIFAT neuron.
     */
    double min_potential_ = default_values<BLIFATNeuron>::min_potential_;
    /**
     * @brief The parameter defines the number of network execution steps, during which the neuron activity is totally
     * blocked.
     */
    int64_t total_blocking_period_ = default_values<BLIFATNeuron>::total_blocking_period_;
    /**
     * @brief The parameter defines a dopamine value used to sum up all incoming dopamine synapse impacts.
     */
    double dopamine_value_ = default_values<BLIFATNeuron>::dopamine_value_;
    /**
     * @brief The parameter defines stochastic stimulation - random namber added to the potential every tick.
     */
    double stochastic_stimulation_ = 0.0;
    /**
     * @brief The random number generator is used for stochastic stimulation.
     */
    unsigned random_number_generator_state_ = 0;
};

}  // namespace knp::neuron_traits
