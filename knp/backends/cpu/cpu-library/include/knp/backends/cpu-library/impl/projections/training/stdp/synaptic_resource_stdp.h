/**
 * @file synaptic_resource_stdp.h
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


namespace knp::backends::cpu::projections::impl::training::stdp
{

template <typename Synapse>
using SynapticResourceSTDPSynapse = knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, Synapse>;


template <typename Synapse>
inline void init_synapse(
    knp::synapse_traits::synapse_parameters<SynapticResourceSTDPSynapse<Synapse>> &params, uint64_t step)
{
    params.rule_.last_spike_step_ = step;
}


template <typename Synapse>
inline void init_projection(
    knp::core::Projection<SynapticResourceSTDPSynapse<Synapse>> &projection,
    std::vector<core::messaging::SpikeMessage> &messages, uint64_t step)
{
}


template <typename Synapse>
inline void modify_weights(knp::core::Projection<SynapticResourceSTDPSynapse<Synapse>> &projection)
{
}


template <typename Synapse>
constexpr bool is_forced(knp::core::Projection<SynapticResourceSTDPSynapse<Synapse>> &projection)
{
    return false;
}

}  //namespace knp::backends::cpu::projections::impl::training::stdp
