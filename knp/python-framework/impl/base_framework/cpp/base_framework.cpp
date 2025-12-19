/**
 * @file base_framework.cpp
 * @brief Python bindings for C++ framework.
 * @kaspersky_support Artiom N.
 * @date 21.02.24
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

#include <knp/framework/backend_loader.h>

#include "../../core/cpp/backend_wrapper.h"
#include "../../core/cpp/population.h"
#include "../../core/cpp/projection.h"
#include "../../core/cpp/uid_utilities.h"
#include "common.h"
#include "exports.h"


template <typename T>
void register_direct_converter()
{
    // Need to register converter.
    // Without this extract from the different module cannot convert Python object to C++ object.
    py::converter::registry::insert(
        [](PyObject *p) { return static_cast<void *>(p); }, py::type_id<T>(),
        &py::converter::registered_pytype_direct<T>::get_pytype);
}


PyObject *PyInit__knp_python_framework_core();

BOOST_PYTHON_MODULE(KNP_FULL_LIBRARY_NAME)
{
    // PyImport_AppendInittab("_knp_python_framework_core", &PyInit__knp_python_framework_base_framework);
    //PyImport_AppendInittab("_knp_python_framework_core", &PyInit__knp_python_framework_core);
    //py::import("knp.core._knp_python_framework_core");
    // auto path_type = py::import("pathlib.Path");
    uid_from_python();
    py::to_python_converter<boost::uuids::uuid, uid_into_python>();
    py::implicitly_convertible<core::UID, boost::uuids::uuid>();

    py::implicitly_convertible<std::string, std::filesystem::path>();
    instance_populations_converters();
    instance_projections_converters();

    //register_direct_converter<knp::core::Backend>();
    //py::register_ptr_to_python<std::shared_ptr<knp::core::Backend>>();
    //py::implicitly_convertible<std::shared_ptr<knp::core::Backend>, Backend>();

    export_input_channel();
    export_model();
    export_model_executor();
    export_model_loader();
    export_network();
    export_network_io();
    export_observers();
    export_output_channel();
}
