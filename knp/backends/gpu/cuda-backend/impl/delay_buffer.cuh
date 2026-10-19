/**
 * @file delay_buffer.cuh
 * @brief Delay buffer used as a part of projection implementation.
 * @kaspersky_support A. Vartenkov.
 * @date 19.10.2026
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

#include <cuda.h>
#include "cuda_lib/vector.cuh"
#include "cuda_lib/safe_call.cuh"

namespace knp::backends::gpu::cuda
{
using LongIndex = unsigned long long;
using BitBufferWord = unsigned int;
constexpr int bits_per_buffer_word = sizeof(BitBufferWord) * 8;


class BitDelayBufferView
{
public:
    BitDelayBufferView(BitBufferWord *data, device_lib::LongIndex num_synapses,
                       device_lib::LongIndex buffer_depth_words)
    : data_(data), num_synapses_(num_synapses), buffer_depth_words_(buffer_depth_words)
    {}


    __device__ void add_impact(device_lib::LongIndex synapse_id,
                               device_lib::LongIndex delay,
                               device_lib::LongIndex step)
    {
        // if delay is too large for the buffer, ignore it. Should never happen unless delays change and buffer doesn't.
        if (delay >= bits_per_buffer_word * buffer_depth_words_) return;
        const unsigned int bit_to_write = (step + delay - 1) % (buffer_depth_words_ * bits_per_buffer_word);
        const unsigned int row = bit_to_write / bits_per_buffer_word;
        const unsigned int shift = bit_to_write % bits_per_buffer_word;
        BitBufferWord &word = data_[row * num_synapses_ + synapse_id];
        word |= BitBufferWord{1} << shift;
    }


    __device__ bool has_impact(device_lib::LongIndex synapse_id, device_lib::LongIndex step)
    {
        const unsigned int bit_to_write = step % (buffer_depth_words_ * bits_per_buffer_word);
        const unsigned int row = bit_to_write / bits_per_buffer_word;
        const unsigned int shift = bit_to_write % bits_per_buffer_word;
        BitBufferWord &word = data_[row * num_synapses_ + synapse_id];
        const BitBufferWord mask = BitBufferWord{1} << shift;
        bool result = word & (mask);
        word &= ~mask;

        return result;
    }

private:
    BitBufferWord *data_;

    device_lib::LongIndex num_synapses_;

    device_lib::LongIndex buffer_depth_words_;
};


class BitDelayBuffer
{
public:
    constexpr static int bits_per_word = sizeof(BitBufferWord) * 8;

    __host__ BitDelayBuffer(device_lib::LongIndex num_synapses, device_lib::LongIndex max_delay)
    : num_synapses_(num_synapses)
    {
        buffer_depth_ = (max_delay + bits_per_word - 1) / bits_per_word;
        BitBufferWord *data;
        LongIndex buffer_size = buffer_depth_ * sizeof(BitBufferWord) * num_synapses;
        device_lib::call_and_check(cudaMalloc(&data, buffer_size));
        device_lib::call_and_check(cudaMemset(data, 0, buffer_size));
        buffer_ = device_lib::CUDAVector(data, num_synapses * buffer_depth_);
    }


    BitDelayBufferView view()
    {
        return BitDelayBufferView{buffer_.data_, num_synapses_, buffer_depth_};
    }

    __host__ __device__ void actualize() { buffer_.actualize(); }

private:
    /**
     * @brief Bitwise flat buffer owner.
     */
    device_lib::CUDAVector<BitBufferWord> buffer_;


    /**
     * @brief buffer length.
     */
    device_lib::LongIndex num_synapses_;


    /**
     * @brief buffer depth, in words. Equals to max delay, divided by bits_per_word and rounded up.
     */
    device_lib::LongIndex buffer_depth_;
};

} // namespace knp::backends::cuda
