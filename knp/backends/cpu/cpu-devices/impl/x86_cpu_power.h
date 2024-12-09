/**
 * @file x86_cpu_power.h
 * @brief CPU power consumption received via Intel PCM library.
 * @kaspersky_support Artiom N.
 * @date 20.02.2023
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

#include <knp/core/impexp.h>

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif

#include <cpucounters.h>

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

#include <chrono>
#include <cinttypes>
#include <vector>


namespace knp::devices::cpu
{

void check_pcm_status(const pcm::PCM::ErrorCode &status);

/**
 * @brief Power meter via Intel PCM.
 */
class KNP_DECLSPEC CpuPower
{
public:
    explicit CpuPower(uint32_t cpu_sock_no);
    float get_power();

private:
    // cppcheck-suppress unusedStructMember
    uint32_t cpu_sock_no_;
    std::chrono::time_point<std::chrono::steady_clock> time_start_;
    pcm::PCM *pcm_instance_ = nullptr;
    // cppcheck-suppress unusedStructMember
    pcm::SocketCounterState sktstate1_, sktstate2_;
    // cppcheck-suppress unusedStructMember
    // std::vector<pcm::CoreCounterState> /* cstates1, */ cstates2_;
    // cppcheck-suppress unusedStructMember
    // pcm::SystemCounterState sstate1_, sstate2_;
};
}  // namespace knp::devices::cpu
