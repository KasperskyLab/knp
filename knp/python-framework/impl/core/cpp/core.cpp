/**
 * @file core.cpp
 * @brief Python bindings for core library.
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

#include <knp/core/core.h>
#include <knp/framework/backend_loader.h>

#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_value_policy.hpp>

#include <filesystem>

#include "any_converter.h"
#include "common.h"
#include "message_endpoint.h"
#include "optional_converter.h"
#include "population.h"
#include "projection.h"
#include "tuple_converter.h"


auto load_backend(knp::framework::BackendLoader& loader, const py::object& backend_path)
{
    return loader.load(py::extract<std::string>(backend_path)());
}


auto make_loader(knp::framework::BackendLoader& self)
{
    return self;
}


BOOST_PYTHON_MODULE(KNP_FULL_LIBRARY_NAME)
{
    spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

    // py::to_python_converter<std::any, to_python_any>();
    // from_python_any();
    py::class_<core::TagMap>(
        "TagMap", "The TagMap class is a definition of tags used by entity and their values.");  // NOLINT
    // .def("get_tag", static_cast<std::any&(core::TagMap::*)(const std::string&)>(&core::TagMap::get_tag),
    //      "Get tag value by tag name.")

    py::class_<core::BaseData>("BaseData", "Common parameters for several different entities.");

    py::implicitly_convertible<core::UID, boost::uuids::uuid>();
    py::implicitly_convertible<std::string, std::filesystem::path>();

    // py::to_python_converter<std::optional<int>, to_python_optional<int>>();

    // Need this for import.
    //    PyObject* sysPath = PySys_GetObject("path");
    //    PyList_Insert(sysPath, 0, PyUnicode_FromString(absolute(std::filesystem::current_path()).string().c_str()));
    //
    //    boost::python::import("libknp_python_framework_neuron_traits");

    py::register_ptr_to_python<std::shared_ptr<knp::core::Backend>>();

    export_backend();
    py::class_<knp::framework::BackendLoader>(
        "BackendLoader", "The BackendLoader class is a definition of a backend loader.")
        .def(
            "__enter__", &make_loader, py::return_self<>(),
            "Make loader.")
        .def(
            "__exit__",
            make_handler([](boost::python::object &self, boost::python::object &exc_type,
                            boost::python::object &exc_value, boost::python::object &traceback) { return false; }),
            "Exit loader.")
        .def("load", &load_backend, "Load backend.")
        .def("is_backend", &knp::framework::BackendLoader::is_backend, "Check if the specified path points to a backend.")
        .staticmethod("is_backend");
    export_device();
    export_message_bus();
    export_message_endpoint();
    export_message_header();
    export_populations();
    export_projections();
    export_spike_message();
    export_subscription();
    export_synaptic_impact();
    export_uid();
}
