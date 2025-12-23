/**
 * @file population.cpp
 * @brief Python bindings for Population.
 * @kaspersky_support Artiom N.
 * @date 07.02.2024
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include "population.h"

#include <vector>

#include <boost/mp11.hpp>

#include "common.h"


/*        .def(
        "get_tags", (auto(core::Population<knp::neuron_traits::BLIFATNeuron>::*)()) &core::Population::get_tags,
        "Get tags used by the population.")
    .def( \
        "get_tags", (auto(core::Population::*)()) & core::Population::get_tags, \
        "Get tags used by the population.")
*/


// .def("get_neurons_parameters", &core::Population<neuron_type>::get_neurons_parameters, "Get parameters of all
// neurons in the population.")

// .def("set_neuron_parameters", static_cast<void (knp::core::Population<neuron_type>::*)(size_t, const
// core::Population<neuron_type>::NeuronParameters&)>(&core::Population<neuron_type>::set_neuron_parameters),
//     "Set parameters for the specific neuron in the population.")

// .def("get_neuron_parameters", &core::Population<neuron_type>::get_neuron_parameters,
// "Get parameters of the specific neuron in the population.")
// py::class_<nt::neuron_parameters<nt::neuron_type>>(BOOST_PP_STRINGIZE(BOOST_PP_CAT(neuron_type, parameters)));

#define INSTANCE_PY_POPULATION(n, template_for_instance, neuron_type)                                                \
    py::class_<core::Population<nt::neuron_type>>(                                                                    \
        BOOST_PP_STRINGIZE(BOOST_PP_CAT(neuron_type, Population)),                                                    \
                           "The Population class is a container of neurons of the same model.", py::no_init)          \
            .def(py::init<core::Population<nt::neuron_type>::NeuronGenerator, size_t>())                              \
            .def(                                                                                                     \
                "__init__",                                                                                           \
                py::make_constructor(static_cast<std::shared_ptr<core::Population<nt::neuron_type>> (*)(              \
                                         const core::UID &, const py::object &, size_t)>(                             \
                    &population_constructor_wrapper<nt::neuron_type>)),                                               \
                "Construct a population by running a neuron generator.")                                              \
            .def(                                                                                                     \
                "__init__",                                                                                           \
                py::make_constructor(                                                                                 \
                    static_cast<std::shared_ptr<core::Population<nt::neuron_type>> (*)(const py::object &, size_t)>(  \
                        &population_constructor_wrapper<nt::neuron_type>)),                                           \
                "Construct a population by running a neuron generator.")                                              \
            .def("add_neurons", &population_neurons_add_wrapper<nt::neuron_type>, "Add neurons to the population.")   \
            .def(                                                                                                     \
                "remove_neurons", &core::Population<nt::neuron_type>::remove_neurons,                                 \
                "Remove neurons with given indexes from the population.")                                             \
            .def(                                                                                                     \
                "remove_neuron", &core::Population<nt::neuron_type>::remove_neuron,                                   \
                "Remove a specific neuron from the population.")                                                      \
            .def("__iter__", py::iterator<core::Population<nt::neuron_type>>(), "Get an iterator of the population.") \
            .def("__len__", &core::Population<nt::neuron_type>::size, "Count number of neurons in the population.")   \
            .def(                                                                                                     \
                "__getitem__",                                                                                        \
                static_cast<core::Population<nt::neuron_type>::NeuronParameters &(                                    \
                    core::Population<nt::neuron_type>::*)(size_t)>(&core::Population<nt::neuron_type>::operator[]),   \
                py::return_internal_reference<>(), "Get parameter values of a neuron with the given index.")          \
            .add_property(                                                                                            \
                "uid",                                                                                                \
                make_handler([](core::Population<nt::neuron_type> &population) { return population.get_uid(); }),     \
                "Get population UID.");


void export_populations()
{
    BOOST_PP_SEQ_FOR_EACH(INSTANCE_PY_POPULATION, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_NEURONS))
    instance_populations_converters();
}