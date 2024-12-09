/**
 * @file cpu.h
 * @brief Class definition for CPU device.
 * @kaspersky_support Artiom N.
 * @date 30.01.2023
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

#include <knp/core/device.h>
#include <knp/core/impexp.h>

#include <memory>
#include <string>
#include <vector>


/**
 * @namespace knp::devices
 * @brief Namespace for implementation of concrete devices.
 */

/**
 * @brief CPU device namespace.
 */
namespace knp::devices::cpu
{

/**
 * @brief The CpuPower class stores the power counter monitor (PCM) instance that monitors the state of a processor.
 */
class CpuPower;

/**
 * @brief The CPU class is a definition of an interface to the CPU device.
 */
class KNP_DECLSPEC CPU : public knp::core::Device  // cppcheck-suppress class_X_Y
{
public:
    /**
     * @brief Avoid copy of a CPU device.
     */
    CPU(const CPU &) = delete;

    /**
     * @brief Avoid copy assignment of a CPU device.
     */
    CPU &operator=(const CPU &) = delete;

    /**
     * @brief CPU device move constructor.
     */
    CPU(CPU &&);

    /**
     * @brief CPU device move operator.
     * @return reference to CPU instance.
     */
    CPU &operator=(CPU &&) noexcept;

    /**
     * @brief CPU device destructor.
     */
    ~CPU() override;

public:
    /**
     * @brief Get device type.
     * @return device type.
     */
    [[nodiscard]] knp::core::DeviceType get_type() const override;

    /**
     * @brief Get device name.
     * @return device name in the arbitrary format.
     */
    [[nodiscard]] const std::string &get_name() const override;

    /**
     * @brief Get CPU device socket number.
     * @return socket number.
     */
    [[nodiscard]] uint32_t get_socket_number() const;

    /**
     * @brief Get power consumption details for the device.
     * @return amount of consumed power.
     */
    [[nodiscard]] float get_power() const override;

private:
    /**
     * @brief CPU device constructor.
     */
    explicit CPU(uint32_t cpu_num);
    friend KNP_DECLSPEC std::vector<CPU> list_processors();

private:
    uint32_t cpu_num_;
    // Non const, because of move operator.
    // cppcheck-suppress unusedStructMember
    std::string cpu_name_;
    mutable std::unique_ptr<CpuPower> power_meter_;
};


/**
 * @brief List all processors on which backend can be initialized.
 * @return vector of CPUs.
 */
KNP_DECLSPEC std::vector<CPU> list_processors();

}  // namespace knp::devices::cpu
