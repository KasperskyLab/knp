/**
 * @file stdp_add_rule.h
 * @brief Rule for additive STDP.
 * @kaspersky_support Artiom N.
 * @date 09.08.2023
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

#include <cinttypes>
#include <vector>

#include "stdp_common.h"


/**
 * @brief Namespace for synapse traits.
 */
namespace knp::synapse_traits
{
/**
 * @brief STDP additive rule parameters.
 * 
 * @note Parameters for the `W(x)` function by Zhang et al. 1998.
 */
template <typename SynapseType>
struct STDPAdditiveRule
{
    /**
     * @brief Type of the synapse linked with rule.
     */
    using LinkedSynapseType = SynapseType;

    /**
     * @brief Time constant in milliseconds intended to increase the weight.
     */
    float tau_plus_ = 10;

    /**
     * @brief Time constant in milliseconds intended to decrease the weight.
     */
    float tau_minus_ = 10;

    /**
     * @brief Indexes of network execution steps on which spikes on the synapse were generated.
     */
    std::vector<uint64_t> presynaptic_spike_times_;

    /**
     * @brief Indexes of network execution steps on which spikes on the axon were generated.
     */
    std::vector<uint64_t> postsynaptic_spike_times_;
};

}  // namespace knp::synapse_traits
