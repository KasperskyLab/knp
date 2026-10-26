/**
 * @file resource_blifat_population_impl.cu
 * @brief Contains functions for Resource BLIFAT population calculation.
 * @kaspersky_support A. Vartenkov
 * @date 17.08.2026
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
#include "../backend_impl.cuh"
#include "../cuda_lib/vector.cuh"
#include "../cuda_lib/register_type.cuh"
#include "../population.cuh"
#include "../projection.cuh"
#include "../resource_delta_projection.cuh"
#include "../populations_impl.cuh"
#include "resource_stdp_impl.cuh"
#include <knp/core/population.h>

#include <vector>

/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
using ResourceBlifatParams = knp::neuron_traits::neuron_parameters<
        knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;
using ResourceSynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using SynapseValue = typename CUDAProjection<ResourceSynapseType>::Synapse;
using SynapseValuePtr = SynapseValue*;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapseType>;

__host__ SynapsesPerNeurons initialize_synapses_per_neurons(const device_lib::IndexView &synapse_index,
                                                            SynapseValue *start);

inline __device__ device_lib::CUDAVectorMutableView<SynapseValue*>
    extract_synapses_from_index(SynapsesPerNeuronsView &synapse_index, device_lib::LongIndex neuron_index)
{
    assert(synapse_index.offsets_size_ > 0);
    device_lib::LongIndex offset = synapse_index.offsets_[neuron_index];
    device_lib::LongIndex size = synapse_index.offsets_[neuron_index + 1] - offset;
#if (!defined(NDEBUG) && defined(FULL_DEBUG))
    printf("Returning synapses: size %llu offset %llu, total size %llu\n", size, offset, synapse_index.synapses_size_);
#endif // NDEBUG
    return device_lib::CUDAVectorMutableView<SynapseValue*>{synapse_index.synapses_ + offset, size};
}

__device__ void recalculate_synapse_weight(ResourceSynapseParams &synapse);


void update_rstdp_synapses(CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
                           SynapsesPerNeuronsView synapses, const device_lib::CUDAVector<SpikeIndex> &spiking_neurons,
                           StepIndex step);
} // namespace knp::backends::gpu::cuda
