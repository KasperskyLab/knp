/**
 * @file interface.h
 * @kaspersky_support Postnikov D.
 * @date 10.12.2025
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

#include <mutex>
#include <unordered_map>
#include <vector>

#include "delta/interface.h"

namespace knp::backends::cpu::projections::impl
{

template <typename Synapse>
MessageQueue::const_iterator calculate_projection_interface(
    knp::core::Projection<Synapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    MessageQueue &future_messages, size_t step_n)
{
    throw std::runtime_error("Unsupported synapse type");
}

template <class Synapse>
void calculate_projection_multithreaded_interface(
    knp::core::Projection<Synapse> &projection, const std::unordered_map<knp::core::Step, size_t> &message_in_data,
    MessageQueue &future_messages, uint64_t step_n, size_t part_start, size_t part_size, std::mutex &mutex)
{
    throw std::runtime_error("Unsupported synapse type");
}

}  //namespace knp::backends::cpu::projections::impl
