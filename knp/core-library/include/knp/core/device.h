/**
 * @file device.h
 * @brief Class definition for device base class.
 * @kaspersky_support Artiom N.
 * @date 26.01.2023
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

#pragma once

#include <knp/core/core.h>

#include <string>

/**
 * @brief Core library namespace.
 */
namespace knp::core
{

/**
 * @brief The DeviceType class defines IDs of supported device types.
 */
enum class DeviceType
{
    /**
     * @brief CPU device.
     */
    CPU,
    /**
     * @brief GPU device.
     */
    GPU,
    /**
     * @brief Generic NPU device.
     */
    NPU
};


/**
 * @brief The Device class is the base class for supported devices.
 */
class Device
{
public:
    /**
     * @brief Device destructor.
     */
    virtual ~Device() = default;

public:
    /**
     * @brief Get device UID.
     * @return device UID.
     */
    [[nodiscard]] const UID &get_uid() const { return base_.uid_; }

    /**
     * @brief Get tags used by the device.
     * @return device tag map.
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

public:
    /**
     * @brief Get device type.
     * @return device type.
     */
    [[nodiscard]] virtual DeviceType get_type() const = 0;

    /**
     * @brief Get device name.
     * @return device name in the arbitrary format.
     */
    [[nodiscard]] virtual const std::string get_name() const = 0;

    /**
     * @brief Get power consumption details for the device.
     * @return amount of consumed power.
     */
    [[nodiscard]] virtual float get_power() const = 0;

protected:
    /**
     * @brief Device base data.
     */
    BaseData base_;
};

}  // namespace knp::core
