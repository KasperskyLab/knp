/**
 * @file dispatch.h
 * @kaspersky_support Postnikov D.
 * @date 21.01.2026
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

#include "additive.h"
#include "synaptic_resource.h"


namespace knp::backends::cpu::projections::impl::training::stdp
{

template <class Synapse>
void init_synapse(knp::synapse_traits::synapse_parameters<Synapse> &params, uint64_t step)
{
}


template <class Synapse>
void init_projection(
    knp::core::Projection<Synapse> &projection, std::vector<core::messaging::SpikeMessage> &messages, uint64_t step)
{
}


template <class Synapse>
void modify_weights(knp::core::Projection<Synapse> &projection)
{
}


template <class Synapse>
constexpr bool is_forced(knp::core::Projection<Synapse> &projection)
{
    return true;
}

}  //namespace knp::backends::cpu::projections::impl::training::stdp
