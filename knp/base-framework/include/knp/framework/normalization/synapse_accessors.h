/**
 * @file synapse_accessors.h
 * @brief Synapse parameter accessor functions.
 * @kaspersky_support Artiom N.
 * @date 23.07.2025
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

#include <knp/core/projection.h>
#include <knp/framework/normalization/normalizers.h>
#include <knp/synapse-traits/all_traits.h>


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{

/**
 * @brief The `WeightAccessor` class is a definition of an interface that provides access to synapse weight parameter (`weight_`).
 * @tparam SynapseType type of the synapse to access.
 */
template <typename SynapseType>
struct WeightAccessor
{
    /**
     * @brief Type of synapse parameters.
     */
    using SynapseParametersType = typename knp::core::Projection<SynapseType>::SynapseParameters;
    /**
     * @brief Synapse weight type.
     */
    using WeightType = decltype(SynapseParametersType::weight_);

    /**
     * @brief Get synapse weight.
     * @param syn_params synapse parameters.
     * @return value of the synapse weight.
     */
    WeightType operator()(const SynapseParametersType &syn_params) const { return syn_params.weight_; }

    /**
     * @brief Correct the synapse weight.
     * @param syn_params synapse parameters.
     * @param weight new synapse weight value to set.
     */
    void operator()(SynapseParametersType &syn_params, const WeightType &weight) const { syn_params.weight_ = weight; }
};

}  // namespace knp::framework::normalization
