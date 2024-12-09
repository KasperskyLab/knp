/**
 * @file cpu_power.cpp
 * @brief CPU device class implementation.
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

#include "x86_cpu_power.h"

#include <spdlog/spdlog.h>
#include <utils.h>

#include <exception>
#include <utility>


namespace knp::devices::cpu
{

void check_pcm_status(const pcm::PCM::ErrorCode& status)
{
    switch (status)
    {
        case pcm::PCM::Success:
        {
            SPDLOG_TRACE("PCM instance programming is ok.");
            break;
        }
        case pcm::PCM::MSRAccessDenied:
            throw std::logic_error(
                "Access to Intel(r) Performance Counter Monitor was denied (no MSR or PCI CFG space access).");
        case pcm::PCM::PMUBusy:
            throw std::logic_error(
                "Access to Intel(r) Performance Counter Monitor was denied (Performance Monitoring Unit is occupied by "
                "other application). Try to stop the application that uses PMU.");
        default:
            throw std::logic_error("Access to Intel(r) Performance Counter Monitor was denied (Unknown error).");
    }
}


CpuPower::CpuPower(uint32_t cpu_sock_no) : cpu_sock_no_(cpu_sock_no), pcm_instance_(pcm::PCM::getInstance())
{
    assert(pcm_instance_);
    // This is an instance, memory releasing is not required.
    const pcm::PCM::ErrorCode status = pcm_instance_->program(pcm::PCM::DEFAULT_EVENTS, nullptr, true, ::getpid());

    try
    {
        check_pcm_status(status);
        get_power();
    }
    catch (const std::logic_error& e)
    {
        SPDLOG_WARN("Error during power consumption counter initialization: {}.", e.what());
    }
}


float CpuPower::get_power()
{
    if (!pcm_instance_->packageEnergyMetricsAvailable())
    {
        throw std::logic_error("Energy metrics are unavailable.");
    }

    std::swap(sktstate1_, sktstate2_);

    sktstate1_ = pcm_instance_->PCM::getSocketCounterState(cpu_sock_no_);
    // pcm_instance_->getAllCounterStates(sstate2_, sktstate2_, cstates2_);
    const auto time_now = std::chrono::steady_clock::now();
    const auto consumed_joules = pcm::getConsumedJoules(sktstate2_, sktstate1_);
    const auto consumed_watts =
        consumed_joules / std::chrono::duration_cast<std::chrono::seconds>(time_now - time_start_).count();

    SPDLOG_DEBUG("CPU, Joules = {}, Watts = {}.", consumed_joules, consumed_watts);

    time_start_ = time_now;

    return consumed_watts;
}
}  // namespace knp::devices::cpu
