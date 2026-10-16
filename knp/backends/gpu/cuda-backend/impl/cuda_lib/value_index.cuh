/**
 * @file value_index.cuh
 * @brief Helper structure for searching.
 * @kaspersky_support A. Vartenkov
 * @date 29.04.2026
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
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

#include "vector.cuh"
#include "../cuda_bus/messaging.cuh"


/**
 * @brief namespace for CUDA functions implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

using LongIndex = unsigned long long;

/**
 * @brief Plain Old Data view structure for a ValueIndex object.
 */
struct IndexView
{
    const LongIndex *indices_ptr_;
    const LongIndex indices_size_;
    const LongIndex *offsets_ptr_;
    const LongIndex offsets_size_;
};


/**
 * @brief Structure to store indexes and offsets.
 * @note it's used to find the connected synapses by a neuron index. You find the offset by indexing the offsets with a
 * neuron's index, and then the number of synapses by difference between this offset and the following one. The part of
 * the indices vector between those offsets gives you the indexes of the connected synapses.
 */
struct ValueIndex
{
    /**
     * @brief Construct a POD view.
     * @return POD structure that can then be sent to __global__ functions easily.
     */
    __host__ __device__ IndexView view() const
    {
        return {indices_.data(), indices_.size(), offsets_.data(), offsets_.size()};
    }

    /**
     * @brief Add a new synapse to an existing index.
     * @param sender connected neuron.
     * @param index synapse index.
     * @note not implemented yet.
     */
    void insert(LongIndex sender, LongIndex index)
    {
        throw std::logic_error("Not implemented");
        // TODO Add insert
    }

    /**
     * @brief Add a new synapse to an existing index.
     * @param sender connected neuron.
     * @param index synapse index.
     * @note not implemented yet.
     */
    void remove(LongIndex sender, LongIndex index)
    {
        throw std::logic_error("Not implemented");
        // TODO Add remove
    }

    /**
     * @brief actualize function, used to turn a shallow copy to a deep copy. See vector.cuh.
     */
    __host__ __device__ void actualize()
    {
        indices_.actualize();
        offsets_.actualize();
    }

    device_lib::CUDAVector<LongIndex> indices_;
    device_lib::CUDAVector<LongIndex> offsets_;
};


/**
 * @brief Calculates the total number of values for all impulses.
 * @param index The built value index.
 * @param inputs the impulses to be use.
 * @return total number of values for the inputs.
 */
__host__ LongIndex count_values_by_indexes(const ValueIndex &index, const CUDAVectorView<cuda::SpikeIndex> inputs);


/**
 * @brief Build exclusive prefix sum for number of synapses per neuron.
 * @param index the built value index.
 * @param inputs spiked neuron indices.
 * @return a vector of "output" offsets for each neuron.
 */
__host__ CUDAVector<LongIndex> calculate_neuron_scan(const ValueIndex &index,
                                                     const CUDAVectorView<cuda::SpikeIndex> inputs);


template <class SynapseType>
__host__ ValueIndex build_index(const knp::core::Projection<SynapseType> &cpu_projection)
{
    // Build map-based index
    std::map<LongIndex, std::vector<LongIndex>> buffer;
    for (size_t i = 0; i < cpu_projection.size(); ++i)
    {
        const auto &synapse = cpu_projection[i];
        const auto neuron_id = std::get<core::source_neuron_id>(synapse);
        auto map_iter = buffer.find(neuron_id);
        if (map_iter == buffer.end())
        {
            map_iter = buffer.insert(std::make_pair(neuron_id, std::vector<LongIndex>{})).first;
        }
        map_iter->second.push_back(i);
    }
    LongIndex current_offset = 0;
    const LongIndex last_neuron = buffer.rbegin()->first;

    std::vector<LongIndex> indices(cpu_projection.size());
    std::vector<LongIndex> offsets;
    offsets.reserve(last_neuron + 2);  // Neuron number + 1 is last_index + 2.
    offsets.push_back(0);
    LongIndex first_neuron = 0;
    for (auto iter = buffer.begin(); iter != buffer.end(); ++iter)
    {
        const LongIndex current_neuron = iter->first;
        // Filling offsets for skipped neurons: if first three are missing that would be 0, 0, 0, 0, 5...
        for (auto i = first_neuron; i < current_neuron; ++i)
        {
            offsets.push_back(current_offset);
        }
        first_neuron = current_neuron + 1;
        std::copy(iter->second.begin(), iter->second.end(), indices.begin() + current_offset);
        current_offset += iter->second.size();
        offsets.push_back(current_offset);
    }
    ValueIndex index;
    index.indices_ = indices;
    index.offsets_ = offsets;
    return index;
}

}  // namespace knp::backends::gpu::cuda::device_lib
