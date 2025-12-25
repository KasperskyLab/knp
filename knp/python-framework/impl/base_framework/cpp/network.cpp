/**
 * @file network.cpp
 * @brief Python bindings for Network class.
 * @kaspersky_support Vartenkov A.
 * @date 05.06.2024
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
#include "network_utility.h"


void export_network()
{
    py::class_<knp::framework::Network>(
        "Network", "The Network class is a definition of a neural network that contains populations and projections.")
        .def(py::init<const knp::core::UID&>())
        .def("add_projection", network_add_projection_variant, "Add a projection to the network.")
        .def("add_population", network_add_population_variant, "Add a population to the network.")
        .def("get_population", &network_get_population_variant, "Get a population with the given UID from the network.")
        .def("get_projection", &network_get_projection_variant, "Get a projection with the given UID from the network.")
        .def(
            "remove_population", &knp::framework::Network::remove_population,
            "Remove a population with the given UID from the network.")
        .def(
            "remove_projection", &knp::framework::Network::remove_projection,
            "Remove a projection with the given UID from the network.")
        .add_property(
            "populations_count", &knp::framework::Network::populations_count, "Count populations in the network.")
        .add_property(
            "projections_count", &knp::framework::Network::projections_count, "Count projections in the network.")
        .def("get_uid", &get_entity_uid<knp::framework::Network>, "Get network UID.")
        .def(
            "populations_range", py::range(&network_begin_populations, &network_end_populations),
            "Get populations range.")
        .def(
            "projections_range", py::range(&network_begin_projections, &network_end_projections),
            "Get projections range.");
}
