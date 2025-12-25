/**
 * @file model_loader.cpp
 * @brief Python bindings for ModelLoader class.
 * @kaspersky_support Vartenkov A.
 * @date 30.10.2024
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
#include "model_loader_utility.h"


void export_model_loader()
{
    py::def("make_model_loader", &make_model_loader, "Create model loader.");

    py::class_<knp::framework::ModelLoader, boost::noncopyable>(
        "ModelLoader",
        "The ModelLoader class is a definition of a loader that uploads the model to the specified backend.",
        py::no_init)
        .def("load", &knp::framework::ModelLoader::load, "Write model to backend.")
        .def(
            "get_inputs", &knp::framework::ModelLoader::get_inputs, py::return_internal_reference<>(),
            "Get input channels.")
        .def(
            "get_outputs", &knp::framework::ModelLoader::get_outputs, py::return_internal_reference<>(),
            "Get output channels.")
        .def(
            "get_input_channel", get_model_loader_input_channel, py::return_internal_reference<>(),
            "Get input channel reference.")
        .def(
            "get_output_channel", get_model_loader_output_channel, py::return_internal_reference<>(),
            "Get output channel reference.");
}
