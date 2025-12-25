/**
 * @file message_bus.cpp
 * @brief Python bindings for message bus.
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

void export_message_bus()
{
    py::class_<core::MessageBus, boost::noncopyable>(
        "MessageBus", "The MessageBus class is a definition of an interface to a message bus.", py::no_init)
        .def(
            "create_endpoint",
            make_handler([](core::MessageBus &self)
                         { return std::make_shared<core::MessageEndpoint>(self.create_endpoint()); }),
            "Create a new endpoint that sends and receives messages through the message bus.")
        .def("step", &core::MessageBus::step, "Route some messages.")
        .def("route_messages", &core::MessageBus::route_messages, "Route messages.");
}