/**
 * @file spike_message.cuh
 * @brief Spike message class.
 * @kaspersky_support Artiom N.
 * @date 02.04.2025
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

#include <knp/core/messaging/messaging.h>

// #include <thrust/device_vector.h>
#include "../cuda_lib/vector.cuh"
#include "../cuda_lib/extraction.cuh"
#include "message_header.cuh"
#include "../uid.cuh"


/**
 * @brief Messaging namespace.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief Spike index type in the form of a 32-bit unsigned integer.
 */
using SpikeIndex = uint32_t;


/**
 * @brief List of spike indexes.
 */
using SpikeData = knp::backends::gpu::cuda::device_lib::CUDAVector<SpikeIndex>;


/**
 * @brief Structure of the spike message.
 */
class SpikeMessage
{
public:
    /**
     * @brief Message header.
     */
    cuda::MessageHeader header_;

    /**
     * @brief Indexes of the recently spiked neurons.
     */
    SpikeData neuron_indexes_;

    /**
     * @todo Maybe add operator `[]` and others to be able to use templates for message processing.
     */

    /**
     * @brief Restore to working order after cudaMemcpy.
     */
     __host__ __device__ void actualize() { neuron_indexes_.actualize(); }

     __host__ __device__ bool operator==(const SpikeMessage &other) const
     {
         return header_ == other.header_ && neuron_indexes_ == other.neuron_indexes_;
     }

     __host__ __device__ bool operator!=(const SpikeMessage &other) const
     {
        return !(*this == other);
     }

};

}  // namespace knp::backends::gpu::cuda
