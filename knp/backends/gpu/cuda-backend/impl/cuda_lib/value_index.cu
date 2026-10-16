/**
 * @file value_index.cu
 * @brief Implementation for a helper structure used for searching.
 * @kaspersky_support A. Vartenkov
 * @date 15.05.2026
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

#include "value_index.cuh"

#include <thrust/transform.h>


namespace knp::backends::gpu::cuda::device_lib
{
__global__ void summarize_index_kernel(IndexView index, device_lib::CUDAVectorView<cuda::SpikeIndex> senders,
                                       LongIndex *result)
{
    const LongIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (index.offsets_size_ == 0 || thread_id >= index.offsets_size_ - 1)
        return;

    LongIndex value = index.offsets_ptr_[thread_id + 1] - index.offsets_ptr_[thread_id];
    atomicAdd(result, value);
}


/**
 * @brief Calculates the total number of values for all impulses.
 * @param index The built value index
 * @param inputs the impulses to be use
 * @return the number.
 */
__host__ LongIndex count_values_by_indexes(const ValueIndex &index,
                                                    const CUDAVectorView<cuda::SpikeIndex> inputs)
{
    SPDLOG_DEBUG("Count values by indexes");
    auto [num_blocks, num_threads] = get_blocks_config(inputs.size_);
    LongIndex *result;
    call_and_check(cudaMalloc(&result, sizeof(LongIndex)));
    call_and_check(cudaMemset(result, 0, sizeof(LongIndex)));
    summarize_index_kernel<<<num_blocks, num_threads>>>(index.view(), inputs, result);
    LongIndex out_result;
    call_and_check(cudaMemcpy(&out_result, result, sizeof(LongIndex), cudaMemcpyDeviceToHost));
    cudaFree(result);
    return out_result;
}


// For each neuron index we find the number of connected synapses and write it to the result.
// Output of the same size as number of neurons
__global__ void gather_index_neuron_kernel(const IndexView index, const CUDAVectorView<cuda::SpikeIndex> inputs,
                                           LongIndex *result)
{
    auto spike_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (spike_index > inputs.size_) return;
    SpikeIndex neuron = inputs.data_[spike_index];
    if (neuron > index.offsets_size_)
        result[spike_index] = 0; // Neuron out of index bounds: no connected synapses
    result[spike_index] = index.offsets_ptr_[neuron + 1] - index.offsets_ptr_[neuron];
}


// For each neuron id in "inputs" find the total number of synapses for "previous" neurons in "inputs".
// Output size() equals to index.size_
__host__ CUDAVector<LongIndex> calculate_neuron_scan(const ValueIndex &index,
                                                     const CUDAVectorView<cuda::SpikeIndex> inputs)
{
    CuMallocAllocator<LongIndex> allocator;
    LongIndex *buffer = allocator.allocate(inputs.size_);
    auto [num_blocks, num_threads] = get_blocks_config(inputs.size_);
    gather_index_neuron_kernel<<<num_blocks, num_threads>>>(index.view(), inputs, buffer);
    // in-place prefix sum, for each neuron the value is the number of synapses before this, starts with 0.
    thrust::exclusive_scan(thrust::device, buffer, buffer + inputs.size_, buffer);
    return CUDAVector<LongIndex>{buffer, inputs.size_}; // The vector would take care of releasing
}
} // namespace knp::backends::gpu::cuda::device_lib
