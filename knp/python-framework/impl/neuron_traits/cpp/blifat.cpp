/**
 * @file blifat.cpp
 * @brief Python bindings for BLIFAT neuron.
 * @kaspersky_support Artiom N.
 * @date 20.02.24
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

#include "common.h"

using bn_params = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;


void export_blifat()
{
    py::class_<bn_params>("BLIFATNeuronParameters", "Structure for BLIFAT neuron parameters")
        .def(py::init<>())
        .add_property(
            "n_time_steps_since_last_firing", &bn_params::n_time_steps_since_last_firing_,
            "The parameter defines a number of network steps since the last spike.")
        .add_property(
            "activation_threshold", &bn_params::activation_threshold_,
            "The parameter defines a threshold for membrane potential.")
        .add_property(
            "dynamic_threshold", &bn_params::dynamic_threshold_,
            "The parameter defines a dynamic threshold for membrane potential after reaching which a neuron generates "
            "a spike.")
        .add_property(
            "threshold_decay", &bn_params::threshold_decay_,
            "The parameter defines a time constant during which the `dynamic_threshold_` parameter tends to its base "
            "value if nothing happens.")
        .add_property(
            "threshold_increment", &bn_params::threshold_increment_,
            "The parameter defines a value that increases the `dynamic_threshold` value if a neuron generates a "
            "spike.")
        .add_property(
            "postsynaptic_trace", &bn_params::postsynaptic_trace_,
            "The parameter defines a threshold after reaching which a neuron generates spikes.")
        .add_property(
            "postsynaptic_trace_decay", &bn_params::postsynaptic_trace_decay_,
            "The parameter defines a time constant during which the `postsynaptic_trace` parameter tends to zero if "
            "nothing happens.")
        .add_property(
            "postsynaptic_trace_increment", &bn_params::postsynaptic_trace_increment_,
            "The parameter defines a value that increases the `postsynaptic_trace` value if a neuron generates a "
            "spike.")
        .add_property(
            "inhibitory_conductance", &bn_params::inhibitory_conductance_,
            "The parameter defines speed with which a potential tends to the `reversal_inhibitory_potential` value.")
        .add_property(
            "inhibitory_conductance_decay", &bn_params::inhibitory_conductance_decay_,
            "The parameter defines a time constant during which the `inhibitory_conductance` value decreases.")
        .add_property("potential", &bn_params::potential_, "The parameter defines the current membrane potential.")
        .add_property(
            "pre_impact_potential", &bn_params::pre_impact_potential_,
            "This parameter is used if there was a blocking signal. If used, all potential changes due to synapses are "
            "ignored.")
        .add_property(
            "potential_decay", &bn_params::potential_decay_,
            "The parameter defines a time constant during which the `potential` value tends to zero.")
        .add_property(
            "bursting_phase", &bn_params::bursting_phase_,
            "The parameter defines a counter for the `bursting_period` value.")
        .add_property(
            "bursting_period", &bn_params::bursting_period_,
            "The parameter defines a number of network steps after reaching which a neuron generates a spike.")
        .add_property(
            "reflexive_weight", &bn_params::reflexive_weight_,
            "The parameter defines a value that increases the membrane potential after a neuron generates a spike.")
        .add_property(
            "reversal_inhibitory_potential", &bn_params::reversal_inhibitory_potential_,
            "The parameter takes the default value of `reversal_inhibitory_potential` defined for a BLIFAT neuron.")
        .add_property(
            "absolute_refractory_period", &bn_params::absolute_refractory_period_,
            "The parameter defines a minimum number of network steps before a neuron can generate the next spike.")
        .add_property(
            "potential_reset_value", &bn_params::potential_reset_value_,
            "The parameter defines a potential value after a neuron generates a spike.")
        .add_property(
            "min_potential", &bn_params::min_potential_,
            "The parameter takes the default value of `min_potential` defined for a BLIFAT neuron.")
        .add_property(
            "total_blocking_period", &bn_params::total_blocking_period_,
            "The parameter defines the number of network execution steps, during which the neuron activity is totally "
            "blocked.")
        .add_property(
            "dopamine_value", &bn_params::dopamine_value_,
            "The parameter defines a dopamine value used to sum up all incoming dopamine synapse impacts.");
}
