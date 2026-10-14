/**
 * @file synapse_slicing_generators.h
 * @brief Parameters generators that can be used for projection upcasting and simplification.
 * @kaspersky_support A. Vartenkov
 * @date 17.02.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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
#include <knp/synapse-traits/all_traits.h>


/**
 * @brief Namespace that provides utilities for generating synapse parameters.
 */
namespace knp::framework::projection::synapse_generators
{
/**
 * @brief Slice a derived synapse parameter set to its base synapse counterpart.

 * @tparam BaseSynapse base synapse type (simpler).
 * @tparam DerivedSynapse derived synapse type (more complex).

 * @param input_synapse_params parameters of the derived synapse that should be sliced.

 * @return parameters of the base synapse obtained by static casting.
 *
 * @details @p BaseSynapse is a simple synapse type that contains only the core parameters needed for the 
 * simulation. @p DerivedSynapse is a more complex synapse that inherits from the base synapse structure and 
 * adds extra fields. For example, @ref knp::synapse_traits::SynapticResourceSTDPDeltaSynapse publicly derives 
 * from @ref knp::synapse_traits::DeltaSynapse. Consequently, 
 * `synapse_parameters<SynapticResourceSTDPDeltaSynapse>` can be sliced to `synapse_parameters<DeltaSynapse>`
 * object, discarding the extra fields.
 *
 * @note This generator is handy when a projection must be exported to a backend that does not support 
 * the extra fields of the derived synapse.
 */
template <class BaseSynapse, class DerivedSynapse>
constexpr synapse_traits::synapse_parameters<BaseSynapse> slice_synapse(
        const synapse_traits::synapse_parameters<DerivedSynapse> &input_synapse_params)
{
    static_assert(
            std::is_base_of_v<
                    synapse_traits::synapse_parameters<BaseSynapse>,
                    synapse_traits::synapse_parameters<DerivedSynapse>
                    >);
    return static_cast<synapse_traits::synapse_parameters<BaseSynapse>>(input_synapse_params);
}


/**
 * @brief Create a generator that converts derived synapse instances to base synapse instances.
 *
 * @tparam BaseSynapse base synapse type (simpler).
 * @tparam DerivedSynapse derived synapse type (more complex).
 *
 * @param projection projection that holds the derived synapse objects.
 *
 * @details @p BaseSynapse is a simple synapse type that contains only the core parameters needed for the 
 * simulation. @p DerivedSynapse is a more complex synapse that inherits from the base synapse structure and 
 * adds extra fields. For example, @ref knp::synapse_traits::SynapticResourceSTDPDeltaSynapse publicly derives 
 * from @ref knp::synapse_traits::DeltaSynapse. Consequently, 
 * `synapse_parameters<SynapticResourceSTDPDeltaSynapse>` can be sliced to `synapse_parameters<DeltaSynapse>`
 * object, discarding the extra fields.
 *
 * @return a generator (`SynapseGenerator`) that yields base synapse objects for each step.
 */
template <class BaseSynapse, class DerivedSynapse>
typename core::Projection<BaseSynapse>::SynapseGenerator make_slicing_synapse_generator(
        const core::Projection<DerivedSynapse> &projection)
{
    // Capture by value because the generator may outlive the original projection.
    typename core::Projection<BaseSynapse>::SynapseGenerator generator = [projection](size_t step)
            -> std::optional<typename core::Projection<BaseSynapse>::Synapse>
    {
        if (step >= projection.size()) return {};
        const auto &element = projection[step];
        return std::make_tuple(slice_synapse<BaseSynapse, DerivedSynapse>(
                std::get<0>(element)), std::get<1>(element), std::get<2>(element));
    };
    return generator;
}


/**
 * @brief Convert a projection that uses a derived synapse type to a projection that uses the base synapse type.
 *
 * @tparam BaseSynapse base synapse type (simpler).
 * @tparam DerivedSynapse derived synapse type (more complex).
 *
 * @param derived_projection projection containing the derived synapse objects.
 *
 * @return a new `core::Projection<BaseSynapse>` that references the same UID, presynaptic and postsynaptic 
 * populations, but provides a generator that yields base synapse objects.
 *
 * @details @p BaseSynapse is a simple synapse type that contains only the core parameters needed for the 
 * simulation. @p DerivedSynapse is a more complex synapse that inherits from the base synapse structure and 
 * adds extra fields. For example, @ref knp::synapse_traits::SynapticResourceSTDPDeltaSynapse publicly derives 
 * from @ref knp::synapse_traits::DeltaSynapse. Consequently, 
 * `synapse_parameters<SynapticResourceSTDPDeltaSynapse>` can be sliced to `synapse_parameters<DeltaSynapse>`
 * object, discarding the extra fields.
 *
 * @note The returned projection does not own the original synapse data; it relies on the generator to produce 
 * sliced copies on demand.
 */
template <class BaseSynapse, class DerivedSynapse>
core::Projection<BaseSynapse> upcast_projection(const core::Projection<DerivedSynapse> &derived_projection)
{
    return core::Projection<BaseSynapse>{derived_projection.get_uid(), derived_projection.get_presynaptic(),
             derived_projection.get_postsynaptic(),
             make_slicing_synapse_generator<BaseSynapse, DerivedSynapse>(derived_projection),
             derived_projection.size()};
}
} // namespace knp::framework::projection::synapse_generators
