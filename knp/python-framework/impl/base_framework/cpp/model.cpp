/**
 * @file model.cpp
 * @brief Python bindings for Model class.
 * @kaspersky_support Vartenkov A.
 * @date 04.06.2024
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
#ifdef KNP_IN_BASE_FW


py::class_<knp::framework::Model>("Model", "The Model class is a definition of a model.", py::no_init)
    .def("__init__", py::make_constructor(&model_constructor), "Initialize model attributes.")
    .def("add_input_channel", &knp::framework::Model::add_input_channel, "Add an input channel to the network.")
    .def("add_output_channel", &knp::framework::Model::add_output_channel, "Add an output channel to the network.")
    .def("get_uid", &get_entity_uid<knp::framework::Model>, "Get model UID.")
    // .add_property("tags", &knp::framework::Model::get_tags)
    .add_property(
        "network", py::make_function(&get_network_from_model, py::return_value_policy<py::reference_existing_object>()),
        &set_network_to_model, "Network associated with model.")
    .add_property("input_channels", &get_model_input_channels, "Get all input channels UIDs.")
    .add_property("output_channels", &get_model_output_channels, "Get all output channels UIDs.");

#endif  // KNP_IN_BASE_FW
