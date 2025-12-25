/**
 * @file message_header.cpp
 * @brief Python bindings for message header.
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
#include "common.h"


void export_message_header()
{
    py::class_<core::messaging::MessageHeader>("MessageHeader", "Common header for messages.")
        .def_readwrite(
            "sender_uid", &core::messaging::MessageHeader::sender_uid_, "UID of the object that sent the message.")
        .def_readwrite("send_time", &core::messaging::MessageHeader::send_time_, "Index of the network execution step.")
        .def(py::self_ns::str(py::self));
}