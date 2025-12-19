/**
 * @file device.cpp
 * @brief Python bindings for device description.
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

void export_device()
{
    py::enum_<core::DeviceType>("DeviceType")
        // CPU device.
        .value("CPU", core::DeviceType::CPU)
        // GPU device.
        .value("GPU", core::DeviceType::GPU)
        // Generic NPU device.
        .value("NPU", core::DeviceType::NPU);

    //py::class_<core::Device>("Device", "The Device class is the base class for devices supported by the device
    //library.");
}
