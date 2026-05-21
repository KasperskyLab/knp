/**
 * @file stdp_common.h
 * @brief STDP internal common templates.
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

#include <set>

#include "type_traits.h"

/**
 * @brief Namespace for synapse traits.
 */
namespace knp::synapse_traits
{

/**
 * @brief Common template for the all STDP implementations.
 * 
 * @tparam Rule type of learning rule that is applied to the base synapse. See `STDPSynapticResourceRule`.
 * @tparam Synapse type of base synapse.
 * 
 * @note It defines a synapse type that can be used as a wrapper over a base synapse.
 */
template <template <typename> typename Rule, typename Synapse>
struct STDP;


/**
 * @brief Synapse with STDP rule.
 * 
 * @details STDP rule is a template, because sometimes parameters can be different in the
 *  different rule/synapse combinations.
 * @code
   template<>
   struct STDPAdditiveRule<DeltaSynapse>
   {
        using LinkedSynapseType = DeltaSynapse;

        float weight1_;
        float weight2_;
    };
 * @endcode
 */
template <template <typename> typename Rule, typename Synapse>
struct synapse_parameters<STDP<Rule, Synapse>> : public synapse_parameters<Synapse>
{
    /**
     * @brief Specific STDP rule type.
     */
    using RuleType = Rule<Synapse>;
    /**
     * @brief Synapse model type.
     */
    using SynapseType = Synapse;

    /**
     * @brief STDP rule parameters.
     */
    RuleType rule_;
};


/**
 * @brief Common STDP parameters.
 * 
 * @tparam Rule STDP rule.
 * @tparam SynapseType synapse type linked with STDP rule.
 */
template <template <typename> typename Rule, typename SynapseType>
struct shared_synapse_parameters<STDP<Rule, SynapseType>>
{
};

}  // namespace knp::synapse_traits
