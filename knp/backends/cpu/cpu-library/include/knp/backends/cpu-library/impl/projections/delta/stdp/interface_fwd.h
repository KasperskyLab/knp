/**
 * @file interface_fwd.h
 * @kaspersky_support Postnikov D.
 * @date 18.12.2025
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
#include <knp/core/messaging/messaging.h>
#include <knp/core/projection.h>

#include <vector>

namespace knp::backends::cpu::projections::impl::delta::stdp
{
template <class DeltaLikeSynapse>
void init_synapse(knp::synapse_traits::synapse_parameters<DeltaLikeSynapse> &params, uint64_t step) = delete;


template <class DeltaLikeSynapse>
void init_projection(
    knp::core::Projection<DeltaLikeSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    uint64_t step) = delete;


template <class DeltaLikeSynapse>
void modify_weights(knp::core::Projection<DeltaLikeSynapse> &projection) = delete;

template <class DeltaLikeSynapse>
constexpr bool is_forced() = delete;

}  //namespace knp::backends::cpu::projections::impl::delta::stdp
