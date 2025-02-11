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

#include <filesystem>

#include "any_converter.h"
#include "common.h"
#include "message_endpoint.h"
#include "optional_converter.h"
#include "population.h"
#include "projection.h"
#include "spike_message.h"
#include "tuple_converter.h"
#include "uid_utilities.h"

// knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>

// py::object my_tf(size_t index)
//{
//     std::cout << "Index ok = " << index << std::endl;
//     return py::object();
// }

std::string get_py_class_name(const py::object &obj_class)
{
    const std::string class_name = boost::python::extract<std::string>(obj_class.attr("__class__").attr("__name__"));
    if (class_name != "class")
    {
        PyErr_SetString(PyExc_TypeError, "Passed object is not a class.");
        py::throw_error_already_set();
        throw std::runtime_error("Not a class.");
    }

    return boost::python::extract<std::string>(obj_class.attr("__name__"));
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

    // py::to_python_converter<std::optional<int>, to_python_optional<int>>();
    //    Py_Initialize();

    // Need this for import.
    //    PyObject* sysPath = PySys_GetObject("path");
    //    PyList_Insert(sysPath, 0, PyUnicode_FromString(absolute(std::filesystem::current_path()).string().c_str()));
    //
    //    boost::python::import("libknp_python_framework_neuron_traits");

#define KNP_IN_CORE
#include "backend.cpp"                  // NOLINT
#include "device.cpp"                   // NOLINT
#include "message_bus.cpp"              // NOLINT
#include "message_endpoint.cpp"         // NOLINT
#include "message_header.cpp"           // NOLINT
#include "population.cpp"               // NOLINT
#include "projection.cpp"               // NOLINT
#include "spike_message.cpp"            // NOLINT
#include "subscription.cpp"             // NOLINT
#include "synaptic_impact_message.cpp"  // NOLINT
#include "uid.cpp"                      // NOLINT
#undef KNP_IN_CORE
}
