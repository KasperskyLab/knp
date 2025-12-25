/**
 * @file model_executor.h
 * @brief Model executor interface.
 * @kaspersky_support Artiom N.
 * @date 21.04.2023
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

#include "common.h"
#include "model_executor_utility.h"


void export_model_executor()
{
    py::class_<knp::framework::ModelExecutor, boost::noncopyable>(
        "ModelExecutor",
        "The ModelExecutor class is a definition of an executor that runs the uploaded model on the specified backend.",
        py::no_init)
        .def("__init__", py::make_constructor(&create_model_executor), "Construct model executor.")
        .def("start", &start_model_executor, "Start model execution.")
        .def("start", &start_model_executor_predicate, "Start model execution with a predicate.")
        .def("stop", &knp::framework::ModelExecutor::stop, "Stop model execution.")
        .def("add_spike_observer", &add_executor_spike_observer, "Add spike observer to model executor.")
        .def("add_impact_observer", &add_executor_impact_observer, "Add impact message observer to model executor.")
        .def("start_learning", &knp::framework::ModelExecutor::start_learning, "Unlock synapse weights.")
        .def("stop_learning", &knp::framework::ModelExecutor::stop_learning, "Lock synapse weights.")
        .def("get_backend", &knp::framework::ModelExecutor::get_backend, "Get reference of backend object.")
        .def(
            "get_output_channel", &get_output_channel, py::return_value_policy<py::reference_existing_object>(),
            "Get output channel reference.");
}
