/**
 * @file exports.h
 * @brief Export function prototypes for Python bindings.
 * @kaspersky_support Artiom N.
 * @date 05.12.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

void export_backend();
void export_device();
void export_message_bus();
void export_message_endpoint();
void export_message_header();
void export_spike_message();
void export_subscription();
void export_synaptic_impact();
void export_uid();
