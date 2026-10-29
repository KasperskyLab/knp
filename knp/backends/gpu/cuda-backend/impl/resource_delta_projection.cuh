/**
 * @file resource_delta_projection.cuh
 * @brief GPU Synaptic Resource STDP projection implementation.
 * @kaspersky_support A. Vartenkov.
 * @date 24.02.2025
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

#include <knp/synapse-traits/all_traits.h>

#include "projection.cuh"

namespace knp::backends::gpu::cuda
{
using RSTDPDeltaSynapse = synapse_traits::SynapticResourceSTDPDeltaSynapse;
using SynapseValue = CUDAProjectionBase<RSTDPDeltaSynapse>::Synapse;


struct SynapsesPerNeuronsView
{
    device_lib::LongIndex offsets_size_;
    device_lib::LongIndex *offsets_;
    device_lib::LongIndex synapses_size_;
    SynapseValue **synapses_;
};


struct SynapsesPerNeurons
{
    device_lib::CUDAVector<device_lib::LongIndex> offsets_;
    device_lib::CUDAVector<SynapseValue *> synapses_;
    SynapsesPerNeuronsView view() { return {offsets_.size(), offsets_.data(), synapses_.size(), synapses_.data()}; }
};


template <>
struct CUDAProjection<RSTDPDeltaSynapse> : public CUDAProjectionBase<RSTDPDeltaSynapse>
{
    CUDAProjection() { is_locked_ = false; }

    __host__ explicit CUDAProjection(const knp::core::Projection<RSTDPDeltaSynapse> &projection)
        : CUDAProjectionBase<RSTDPDeltaSynapse>(projection)
    {
    }

    /**
     * @note Clear this if you are modifying projection connectivity.
     */
    SynapsesPerNeurons synapses_by_postsynaptic_;
};
}  // namespace knp::backends::gpu::cuda
