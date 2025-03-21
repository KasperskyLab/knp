/**
 * @file resource_stdp_delta.cpp
 * @brief Python bindings for resource stdp delta synapse.
 * @kaspersky_support Vartenkov A.
 * @date 25.10.24
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

#if defined(_KNP_IN_SYNAPSE_TRAITS)
using rds_params = knp::synapse_traits::synapse_parameters<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using rds_rule = knp::synapse_traits::STDPSynapticResourceRule<knp::synapse_traits::DeltaSynapse>;


py::class_<rds_rule>(
    "SynapticResourceSTDPDeltaSynapseRule", "Delta synapse parameters with support of synaptic resource-based STDP.")
    .def(py::init<>())
    .add_property(
        "synaptic_resource", &rds_rule::synaptic_resource_, "Synaptic resource that determines synapse weight.")
    .add_property("min_weight", &rds_rule::w_min_, "Minimal weight value.")
    .add_property("max_weight", &rds_rule::w_max_, "Maximal weight value.")
    .add_property("d_u", &rds_rule::d_u_, "Resource decreasing constant.")
    .add_property(
        "dopamine_plasticity_period", &rds_rule::dopamine_plasticity_period_,
        "Dopamine plasticity period. If a neuron is rewarded during this period, then the synapse weight changes.")
    .add_property("last_spike_step", &rds_rule::last_spike_step_, "Step number of when the synapse got a spike last.")
    .add_property(
        "had_hebbian_update", &rds_rule::had_hebbian_update_,
        "True if the synapse was updated during the current spike sequence.")
    .add_property(
        "had_contributed", &rds_rule::has_contributed_,
        "True if a postsynaptic neuron had a spike recently after this synapse got one.");


py::class_<rds_params, py::bases<ds_params>>(
    "SynapticResourceSTDPDeltaSynapseParameters",
    "Structure for Delta synapse parameters with support of synaptic resource-based STDP.")
    .def(py::init<>())
    .add_property("weight", &rds_params::weight_, "Synaptic weight.")
    .add_property(
        "delay", &rds_params::delay_,
        "Synaptic delay. Delay of `N` means that a spike sent on step `X` will be received on step `X + N`.")
    .add_property(
        "output_type", &rds_params::output_type_,
        "Synapse type. Various types have different influences on neuron parameters as defined by neuron function.")
    .add_property("rule", &rds_params::rule_, "Structure containing synaptic plasticity parameters.");


#endif
