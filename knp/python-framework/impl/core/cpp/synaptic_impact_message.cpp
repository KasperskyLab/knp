/**
 * @file synaptic_impact_message.cpp
 * @brief Python bindings for SynapticImpactMessage.
 * @kaspersky_support Artiom N.
 * @date 01.02.2024
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

namespace msg = knp::core::messaging;

void export_synaptic_impact()
{
    py::class_<msg::SynapticImpact>(
        "SynapticImpact",
        "Structure that contains the synaptic impact value and indexes of presynaptic and postsynaptic neurons.")
        .def_readwrite("connection_index", &msg::SynapticImpact::connection_index_, "Index of the population synapse.")
        .def_readwrite(
            "impact_value", &msg::SynapticImpact::impact_value_, "Value used to change neuron membrane potential.")
        .def_readwrite(
            "synapse_type", &msg::SynapticImpact::synapse_type_,
            "Synapse type that might define the value role inside the neuron function.")
        .def_readwrite(
            "presynaptic_neuron_index", &msg::SynapticImpact::presynaptic_neuron_index_,
            "Index of the presynaptic neuron connected to the synapse.")
        .def_readwrite(
            "postsynaptic_neuron_index", &msg::SynapticImpact::postsynaptic_neuron_index_,
            "Index of the postsynaptic neuron connected to the synapse.");


    py::class_<msg::SynapticImpactMessage>("SynapticImpactMessage", "Structure of the synaptic impact message.")
        .def_readwrite("header", &msg::SynapticImpactMessage::header_, "Message header.")
        .def_readwrite(
            "presynaptic_population_uid", &msg::SynapticImpactMessage::presynaptic_population_uid_,
            "UID of the population that sends spikes to the projection.")
        .def_readwrite(
            "postsynaptic_population_uid", &msg::SynapticImpactMessage::postsynaptic_population_uid_,
            "UID of the population that receives impacts from the projection.")
        .def_readwrite("impacts", &msg::SynapticImpactMessage::impacts_, "Impact values.")
        .def_readwrite(
            "is_forcing", &msg::SynapticImpactMessage::is_forcing_,
            "Boolean value that defines whether the signal is from a projection without plasticity.");
}