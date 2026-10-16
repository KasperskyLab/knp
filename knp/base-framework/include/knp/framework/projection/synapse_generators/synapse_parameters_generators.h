/**
 * @file synapse_parameters_generators.h
 * @brief Parameters generators.
 * @kaspersky_support Artiom N.
 * @date 17.10.2024
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

#include <knp/core/projection.h>

#include <functional>
#include <optional>

/**
 * @brief Projection namespace.
 */
namespace knp::framework::projection
{

/**
 * @brief Namespace for framework projection connectors.
 */
namespace parameters_generators
{

/**
 * @brief Two parameters (index, index) generator type.
 * 
 * @tparam SynapseType synapse type which parameters need to be generated.
 */
template <typename SynapseType>
using SynGen2ParamsType = std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t, size_t)>;


/**
 * @brief One parameter (index) generator type.
 * 
 * @tparam SynapseType synapse type which parameters need to be generated.
 */
template <typename SynapseType>
using SynGen1ParamType = std::function<typename knp::core::Projection<SynapseType>::SynapseParameters(size_t)>;


/**
 * @brief Two parameters (index, index) generator type with optional result.
 * 
 * @tparam SynapseType synapse type which parameters need to be generated.
 */
template <typename SynapseType>
using SynGenOptional2ParamsType =
    std::function<typename std::optional<typename knp::core::Projection<SynapseType>::SynapseParameters>(
        size_t, size_t)>;


/**
 * @brief Generate default synapse parameters for a synapse with the specified index.
 *
 * @tparam SynapseType synapse type for which parameters are generated.
 *
 * @return default‑constructed synapse parameters.
 */
template <typename SynapseType>
typename knp::core::Projection<SynapseType>::SynapseParameters default_synapse_gen_1param(size_t)  // NOLINT
{
    return typename knp::core::Projection<SynapseType>::SynapseParameters();
}


/**
 * @brief Generate default synapse parameters for a synapse that connects neurons with specified indexes.
 *
 * @tparam SynapseType synapse type for which parameters are generated.
 *
 * @return default‑constructed synapse parameters.
 */
template <typename SynapseType>
typename knp::core::Projection<SynapseType>::SynapseParameters default_synapse_gen_2param(size_t, size_t)  // NOLINT
{
    return typename knp::core::Projection<SynapseType>::SynapseParameters();
}


/**
 * @brief The CopySynapseGen class is a definition of a synapse generator that copies parameters of the specified
 * synapse.
 * 
 * @tparam SynapseType synapse type.
 */
template <typename SynapseType>
class CopySynapseGen
{
public:
    /**
     * @brief Type of the synapse parameters.
     */
    using SynapseParametersType = typename knp::core::Projection<SynapseType>::SynapseParameters;

    /**
     * @brief Copy synapse constructor.
     * 
     * @param base_synapse synapse parameters to copy.
     */
    explicit CopySynapseGen(const SynapseParametersType& base_synapse) : base_synapse_(base_synapse) {}

    /**
     * @brief Generation operator.
     * 
     * @return synapse parameters that are the same as the source synapse parameters.
     */
    SynapseParametersType operator()(size_t, size_t) const  // NOLINT
    {
        return base_synapse_;
    }

private:
    SynapseParametersType base_synapse_;
};

}  // namespace parameters_generators

}  // namespace knp::framework::projection
