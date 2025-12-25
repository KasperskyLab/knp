/**
 * @file subscription.cpp
 * @brief Python bindings for Subscription.
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

#include <knp/core/subscription.h>

#include "common.h"


namespace cm = knp::core::messaging;

#define INSTANCE_PY_SUBSCRIPTION(n, template_for_instance, message_type)                                              \
    py::class_<core::Subscription<cm::message_type>>(                                                                 \
        BOOST_PP_STRINGIZE(BOOST_PP_CAT(message_type, Subscription)),                                                 \
                           "The Subscription class is used for message exchange between network entities.",           \
                           py::no_init)                                                                               \
            .def("remove_sender", &core::Subscription<cm::message_type>::remove_sender, "Unsubscribe from a sender.") \
            .def(                                                                                                     \
                "add_sender", &core::Subscription<cm::message_type>::add_sender,                                      \
                "Add a sender with the given UID to the subscription.")                                               \
            .def(                                                                                                     \
                "add_senders", &core::Subscription<cm::message_type>::add_senders,                                    \
                "Add several senders to the subscription.")                                                           \
            .def(                                                                                                     \
                "has_sender", &core::Subscription<cm::message_type>::has_sender,                                      \
                "Check if a sender with the given UID exists.")                                                       \
            .def(                                                                                                     \
                "add_message",                                                                                        \
                static_cast<void (core::Subscription<cm::message_type>::*)(const cm::message_type&)>(                 \
                    &core::Subscription<cm::message_type>::add_message),                                              \
                "Add a message to the subscription.");
// .def(
//     "get_messages",
//     static_cast<core::Subscription<cm::message_type>::MessageContainerType(core::Subscription<cm::message_type>::*)()>(&core::Subscription<cm::message_type>::get_messages),
//     "Get all messages.")

void export_subscription()
{
    BOOST_PP_SEQ_FOR_EACH(INSTANCE_PY_SUBSCRIPTION, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_MESSAGES))
}
