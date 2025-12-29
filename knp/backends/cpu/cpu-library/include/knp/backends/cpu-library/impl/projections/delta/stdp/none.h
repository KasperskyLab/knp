/**
 * @file none.h
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

#include <knp/backends/cpu-library/impl/projections/delta/shared.h>

#include <vector>

#include "interface_fwd.h"

namespace knp::backends::cpu::projections::impl::delta::stdp
{
template <>
inline void init_synapse<DeltaSynapse>(knp::synapse_traits::synapse_parameters<DeltaSynapse> &params, uint64_t step)
{
}

template <>
inline void init_projection<DeltaSynapse>(
    knp::core::Projection<DeltaSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    uint64_t step)
{
}

template <>
inline void modify_weights<DeltaSynapse>(knp::core::Projection<DeltaSynapse> &projection)
{
}

template <>
constexpr bool is_forced<DeltaSynapse>()
{
    return true;
}

}  //namespace knp::backends::cpu::projections::impl::delta::stdp
