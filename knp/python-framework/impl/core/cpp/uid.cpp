/**
 * @file uid.cpp
 * @brief Python bindings for UID.
 * @kaspersky_support Artiom N.
 * @date 05.02.2024
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


#if defined(KNP_IN_CORE)

py_uuid_UUID = py::import("uuid").attr("UUID");

py::implicitly_convertible<core::UID, boost::uuids::uuid>();

py::class_<core::continuously_uid_generator>("continuously_uid_generator", "UID generator for entities.");

py::enum_<boost::uuids::uuid::variant_type>("uuid_variant_type", "UUID variant type.")
    .value("NCS", boost::uuids::uuid::variant_type::variant_ncs)
    .value("RFC_4122", boost::uuids::uuid::variant_type::variant_rfc_4122)
    .value("MICROSOFT", boost::uuids::uuid::variant_type::variant_microsoft)
    .value("FUTURE", boost::uuids::uuid::variant_type::variant_future);

py::class_<boost::uuids::uuid>("uuid", "Boost UUID.")
    .add_property("size", &boost::uuids::uuid::size, "Define UUID size in bytes.")
    .add_property("is_nil", &boost::uuids::uuid::is_nil, "Define if UUID is empty.")
    .add_property("variant", &boost::uuids::uuid::variant, "Define UID format variant.");

py::to_python_converter<boost::uuids::uuid, uid_into_python>();
uid_from_python();

py::class_<core::UID>("UID", "The UID class is a definition of unique identifiers for entities.")
    .def(py::init<bool>(py::arg("random")))     // Generate a random UID.
    .def(py::init<const boost::uuids::uuid>())  // Create a UID from `boost::uuids::uuid`.
    .def(py::init<boost::uuids::uuid>())        // Create a UID from `boost::uuids::uuid`.
    // Create a UID from an array of 16 uint8_t integers.
    .def(py::init<std::array<
             boost::uuids::uuid::value_type,
             sizeof(boost::uuids::uuid::data) / sizeof(boost::uuids::uuid::data[0])>>())
    .def(py::init<core::UID>())  // "Constructor that creates a copy of an existing UID."
    .def(py::self_ns::str(py::self))
    .def("__bool__", &core::UID::operator bool, "Check if UID is valid.")
    // cppcheck-suppress syntaxError
    .def("__lt__", &core::UID::operator<, "Check if the current UID is less than the specified UID.")
    .def("__eq__", &core::UID::operator==, "Check if two UIDs are the same.")
    .def("__ne__", &core::UID::operator!=, "Check if two UIDs are different.")
    .def("__hash__", &get_py_hash, "Hash value for UID.")
    .add_property("tag", &core::UID::tag, "UID value.");

py::class_<core::uid_hash>("uid_hash", "UID hash functor type.")
    .def("__call__", &core::uid_hash::operator(), "Get a hash value of the specified UID.");

// py::class_<std::vector<core::UID>>("UIDs").def(py::vector_indexing_suite<core::UID>());

#endif
