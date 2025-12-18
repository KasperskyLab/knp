/**
 * @file generators.h
 * @brief Common generators used for tests.
 * @kaspersky_support Artiom N.
 * @date 23.06.2023
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

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <optional>


/**
 * @brief Test namespace.
 */
namespace knp::testing
{
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using BLIFATPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;
using ResourceBlifatPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;


// Create an input projection.
inline std::optional<DeltaProjection::Synapse> input_projection_gen(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}

// Create a loop projection.
inline std::optional<DeltaProjection::Synapse> synapse_generator(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.1, 6, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}

// Create an input resource projection
inline std::optional<ResourceDeltaProjection::Synapse> input_res_projection_gen(size_t /*index*/)  // NOLINT
{
    knp::synapse_traits::synapse_parameters<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> syn;
    syn.rule_.w_max_ = 2.0;
    syn.rule_.w_min_ = 1.0;
    syn.rule_.synaptic_resource_ = 1.0;
    syn.rule_.dopamine_plasticity_period_ = 5;
    syn.weight_ = 1.5;
    syn.delay_ = 1;
    syn.output_type_ = synapse_traits::OutputType::EXCITATORY;
    return ResourceDeltaProjection::Synapse{syn, 0, 0};
}

// Create a loop resource projection
inline std::optional<ResourceDeltaProjection::Synapse> loop_res_projection_gen(size_t /*index*/)  // NOLINT
{
    knp::synapse_traits::synapse_parameters<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> syn;
    syn.rule_.w_max_ = 2.0;
    syn.rule_.w_min_ = 1.0;
    syn.rule_.synaptic_resource_ = 1.0;
    syn.rule_.dopamine_plasticity_period_ = 5;
    syn.weight_ = 1.5;
    syn.delay_ = 6;
    syn.output_type_ = synapse_traits::OutputType::EXCITATORY;
    return ResourceDeltaProjection::Synapse{syn, 0, 0};
}


// Create population.
inline knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> neuron_generator(size_t)  // NOLINT
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>{};
}


// Create resource population
inline auto neuron_res_generator(size_t)  // NOLINT
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>{};
}

}  // namespace knp::testing
