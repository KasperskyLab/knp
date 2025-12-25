/**
 * @file resource_stdp_blifat.cpp
 * @brief Python bindings for resource stdp BLIFAT neuron.
 * @kaspersky_support Vartenkov A.
 * @date 28.10.24
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
 */


#include <knp/neuron-traits/all_traits.h>

#include "common.h"


void export_resource_stdp_blifat()
{
    using rsbn_params = knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
    using bn_params = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;


    py::class_<rsbn_params, py::bases<bn_params>>(
        "SynapticResourceSTDPBLIFATNeuronParameters",
        "Structure for BLIFAT neuron parameters with support of synaptic resource-based STDP.")
        .def(py::init<>())
        .def_readwrite(
            "dopamine_plasticity_time", &rsbn_params::dopamine_plasticity_time_,
            "Time parameter for dopamine plasticity.")
        .def_readwrite("free_synaptic_resource", &rsbn_params::free_synaptic_resource_, "Free synaptic resource.")
        .def_readwrite(
            "synaptic_resource_threshold", &rsbn_params::synaptic_resource_threshold_,
            "Synaptic resource threshold value.")
        .def_readwrite(
            "resource_drain_coefficient", &rsbn_params::resource_drain_coefficient_,
            "Synaptic resource divided by `number of synapses + resource drain coefficient`. "
            "The parameter value defines the number of silent synapses.")
        .def_readwrite("stability", &rsbn_params::stability_, "Dynamic synapse attribute for stability.")
        .def_readwrite(
            "stability_change_parameter", &rsbn_params::stability_change_parameter_,
            "The parameter defines the stability fluctuation value.")
        .def_readwrite(
            "stability_change_at_isi", &rsbn_params::stability_change_at_isi_,
            "This value is added to stability at the beginning of each ISI period.")
        .def_readwrite("isi_max", &rsbn_params::isi_max_, "Time between spikes in the ISI period.")
        .def_readwrite("d_h", &rsbn_params::d_h_, "Hebbian plasticity value.")
        .def_readwrite("isi_status", &rsbn_params::isi_status_, "ISI period status.")
        .def_readwrite("last_step", &rsbn_params::last_step_, "Last non-forced spike step.")
        .def_readwrite(
            "first_isi_spike", &rsbn_params::first_isi_spike_,
            "Step of the last unforced spike. Used to update the dopamine value.")
        .def_readwrite(
            "is_being_forced", &rsbn_params::is_being_forced_, "`true` if a neuron receives a forced synaptic signal.");
}
