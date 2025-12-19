/**
 * @file population.cuh
 * @brief CUDA popualtion class.
 * @kaspersky_support Artiom N.
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

#include <knp/core/population.h>
#include <knp/neuron-traits/all_traits.h>

#include "cuda_lib/vector.cuh"
#include "uid.cuh"
#include "cuda_bus/messaging.cuh"


/**
 * @brief Namespace for single-threaded backend.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief The CUDAPopulation class is a definition of a CUDA neurons population.
 */
template <typename NeuronType>
struct CUDAPopulation
{
    /**
     * @brief Type of the population neurons.
     */
    using PopulationNeuronType = NeuronType;
    /**
     * @brief Population of neurons with the specified neuron type.
     */
    using PopulationType = CUDAPopulation<NeuronType>;
    /**
     * @brief Neuron parameters and their values for the specified neuron type.
     */
    using NeuronParameters = ::knp::neuron_traits::neuron_parameters<NeuronType>;

    __host__ __device__ CUDAPopulation() = default;

//#if !defined(__CUDA_ARCH__)
    /**
     * @brief Constructor.
     * @param population source population.
     */
    __host__ explicit CUDAPopulation(const knp::core::Population<NeuronType> &population)
        : uid_{to_gpu_uid(population.get_uid())},
          neurons_{population.get_neurons_parameters()}
    {
    }
//#endif

    __host__ __device__ ~CUDAPopulation() = default;

    __host__ __device__ void actualize() { neurons_.actualize(); }

    /**
     * @brief UID.
     */
    cuda::UID uid_;
    /**
     * @brief Neurons.
     */
    cuda::device_lib::CUDAVector<NeuronParameters> neurons_;
};

} // namespace knp::backends::gpu::cuda
