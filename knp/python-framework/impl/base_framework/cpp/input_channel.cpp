/**
 * @file input_channel.cpp
 * @brief Python bindings for InputChannel.
 * @kaspersky_support Vartenkov A.
 * @date 30.05.2024
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
#include "input_channel_utility.h"


void export_input_channel()
{
    py::class_<knp::framework::io::input::InputChannel, boost::noncopyable>(
        "InputChannel", "The InputChannel class is a definition of an input channel.", py::no_init)
        .def("__init__", py::make_constructor(&construct_input_channel), "Initialize the input channel attributes.")
        .def("get_uid", &get_entity_uid<knp::framework::io::input::InputChannel>, "Get input channel UID.")
        .def(
            "send", &knp::framework::io::input::InputChannel::send,
            "Read data from input stream, form a spike message and send it to an endpoint.");
}
