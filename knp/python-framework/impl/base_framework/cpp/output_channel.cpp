/**
 * @file output_channel.cpp
 * @brief Python bindings for OutputChannel.
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
#include "output_channel_utility.h"


void export_output_channel()
{
    py::class_<knp::framework::io::output::OutputChannel, boost::noncopyable>(
        "OutputChannel", "The OutputChannel class is a definition of an output channel.", py::no_init)
        .def("get_uid", &get_entity_uid<knp::framework::io::output::OutputChannel>, "Get output channel UID.")
        .def(
            "update", &knp::framework::io::output::OutputChannel::update,
            "Unload spike messages from the endpoint into the message buffer.")
        .def(
            "read_some_from_buffer", &read_from_buffer,
            "Read a specified interval of messages from sorted internal message buffer.")
        .def("__init__", &construct_output_channel, "Initialize output channel attributes.");
}
