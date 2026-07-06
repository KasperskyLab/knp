/**
 * @file uid.h
 * @brief Python bindings header for UID.
 * @kaspersky_support Artiom N.
 * @date 22.02.2024
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

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "common.h"


static py::object py_uuid_UUID;


struct uid_into_python
{
    static PyObject* convert(boost::uuids::uuid const& u)
    {
        return py::incref(py_uuid_UUID(py::object(), std::vector<uint8_t>(u.begin(), u.end())).ptr());
    }
};


struct uid_from_python
{
    uid_from_python() { py::converter::registry::push_back(convertible, construct, py::type_id<boost::uuids::uuid>()); }

    static void* convertible(PyObject* obj) { return PyObject_IsInstance(obj, py_uuid_UUID.ptr()) ? obj : nullptr; }

    static void construct(PyObject* obj, py::converter::rvalue_from_python_stage1_data* data)
    {
        py::object bytes_object = py::object(py::handle<>(py::borrowed(obj))).attr("bytes");
        char* bytes = nullptr;
        Py_ssize_t bytes_size = 0;
        if (PyBytes_AsStringAndSize(bytes_object.ptr(), &bytes, &bytes_size) == -1)
        {
            py::throw_error_already_set();
        }

        constexpr auto expected_size = boost::uuids::uuid::static_size();
        if (bytes_size != static_cast<Py_ssize_t>(expected_size))
        {
            const auto error_message = "UUID byte array must contain exactly " + std::to_string(expected_size) +
                                       " bytes, got " + std::to_string(bytes_size) + ".";
            PyErr_SetString(PyExc_ValueError, error_message.c_str());
            py::throw_error_already_set();
        }

        auto storage =
            reinterpret_cast<py::converter::rvalue_from_python_storage<boost::uuids::uuid>*>(data)->storage.bytes;
        boost::uuids::uuid* res = new (storage) boost::uuids::uuid;

        std::copy_n(reinterpret_cast<const uint8_t*>(bytes), expected_size, res->begin());
        data->convertible = storage;
    }
};


inline auto get_py_hash(const knp::core::UID& uid)
{
    const knp::core::uid_hash hash;
    return hash(uid);
}
