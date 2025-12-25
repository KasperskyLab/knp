/**
 * @file observer.cpp
 * @brief Python bindings for MessageObserver.
 * @kaspersky_support Vartenkov A.
 * @date 05.06.2024
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
#include "observer_utility.h"


void export_observers()
{
    using SpikeMsg = knp::core::messaging::SpikeMessage;
    using ImpactMsg = knp::core::messaging::SynapticImpactMessage;
    using SpikeObserver = knp::framework::monitoring::MessageObserver<SpikeMsg>;
    using ImpactObserver = knp::framework::monitoring::MessageObserver<ImpactMsg>;


    py::class_<SpikeObserver, boost::noncopyable>(
        "SpikeMessageObserver",
        "The SpikeMessageObserver class is a definition of an observer that receives spike messages and processes "
        "them.",
        py::no_init)
        .def(
            "__init__", py::make_constructor(&make_observer<SpikeMsg>), "Create an observer to process spike messages.")
        .def("get_uid", &get_entity_uid<SpikeObserver>, "Get observer UID.")
        .def("subscribe", &SpikeObserver::subscribe, "Subscribe to spike messages from a list of entities.");


    py::class_<ImpactObserver, boost::noncopyable>(
        "ImpactMessageObserver",
        "The ImpactMessageObserver class is a definition of an observer that receives synaptic impact messages and "
        "processes them.",
        py::no_init)
        .def(
            "__init__", py::make_constructor(&make_observer<ImpactMsg>),
            "Create an observer to process synaptic impact messages.")
        .def("get_uid", &get_entity_uid<ImpactObserver>, "Get observer UID.")
        .def("subscribe", &ImpactObserver::subscribe, "Subscribe to synaptic impact messages from a list of entities.");
}
