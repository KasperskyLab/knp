/**
 * @file unknown_cpu.cpp
 * @brief Not fully supported CPU device class implementation.
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

#include <knp/devices/cpu.h>

#include <spdlog/spdlog.h>

#include <exception>

#include <boost/uuid/name_generator.hpp>


namespace knp::devices::cpu
{

static constexpr const char* ns_uid = "0000-0000-0000-0000";


/**
 * @brief Fake power meter for unknown CPU.
 */
class KNP_DECLSPEC CpuPower
{
public:
    explicit CpuPower(uint32_t /*cpu_sock_no*/) {}
    float get_power() const { return 0.0; }
};


CPU::CPU(uint32_t cpu_num) : cpu_num_(cpu_num), power_meter_{std::make_unique<CpuPower>(cpu_num)}
{
    cpu_name_ = "Unknown CPU " + std::to_string(cpu_num_);
    Device::base_.uid_ = knp::core::UID(boost::uuids::name_generator(core::UID(ns_uid))(cpu_name_.c_str()));
}


CPU::CPU(CPU&& other) : cpu_name_{std::move(other.cpu_name_)} {}


CPU::~CPU() {}


CPU& CPU::operator=(CPU&& other) noexcept
{
    cpu_name_.swap(other.cpu_name_);
    power_meter_.swap(other.power_meter_);
    return *this;
}


knp::core::DeviceType CPU::get_type() const
{
    return knp::core::DeviceType::CPU;
}


const std::string& CPU::get_name() const
{
    return cpu_name_;
}


uint32_t CPU::get_socket_number() const
{
    return cpu_num_;
}


float CPU::get_power() const
{
    return 0;
}


KNP_DECLSPEC std::vector<CPU> list_processors()
{
    std::vector<CPU> result;

    result.push_back(CPU(0));

    return result;
}

}  // namespace knp::devices::cpu
