/**
 * @file astrocyte_nonlinear_dispatcher.h
 * @brief Dispatch functions for astrocyte state updates.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "astrocyte_nonlinear_impl.h"


namespace knp::backends::cpu::populations::impl
{

/**
 * @brief Calculate the pre-impact state of a single astrocyte.
 * @param astrocyte astrocyte parameters.
 */
inline void calculate_pre_impact_single_neuron_state_dispatch(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteNonlinear> &astrocyte)
{
    astrocyte_nonlinear::calculate_pre_impact_single_neuron_state_impl(astrocyte);
}


/**
 * @brief Apply a synaptic impact to an astrocyte.
 * @param astrocyte astrocyte parameters.
 * @param impact impact message; its weight is ignored.
 * @param is_forcing forcing flag; ignored.
 */
inline void impact_neuron_dispatch(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteNonlinear> &astrocyte,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    astrocyte_nonlinear::impact_neuron_impl(astrocyte, impact, is_forcing);
}


/**
 * @brief Calculate the post-impact state of a single astrocyte.
 * @param astrocyte astrocyte parameters.
 * @return true if calcium reaches or exceeds the modulation threshold.
 */
inline bool calculate_post_impact_single_neuron_state_dispatch(
    knp::astrocyte_traits::astrocyte_parameters<knp::astrocyte_traits::AstrocyteNonlinear> &astrocyte)
{
    return astrocyte_nonlinear::calculate_post_impact_single_neuron_state_impl(astrocyte);
}

}  // namespace knp::backends::cpu::populations::impl
