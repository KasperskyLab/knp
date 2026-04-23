/**
 * @file convert_count.h
 * @brief Header for counting converter.
 * @kaspersky_support Vartenkov Andrey
 * @date 01.06.2023
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

#include <knp/core/impexp.h>
#include <knp/core/messaging/spike_message.h>

#include <vector>


/**
 * @brief Output channel namespace.
 */
namespace knp::framework::io::output
{
/**
 * @brief Convert a set of spike messages to a vector where an element value is equal to the number of times that a
 * neuron with the corresponding index has spiked.
 * 
 * @param message_list list of spike messages that contain indexes of spiked neurons.
 * @param output_size output vector size (usually corresponds to the size of an output population).
 * 
 * @return vector.
 * 
 * @details For example, a method, where `output_size` equals 6 and `message_list` contains messages `{0, 2}`, `{2, 4}`,
 * `{1, 2}`, will return a vector `{1, 1, 3, 0, 1, 0}`.
 */
std::vector<size_t> KNP_DECLSPEC
converter_count(const std::vector<core::messaging::SpikeMessage> &message_list, size_t output_size)
{
    std::vector<size_t> result(output_size, 0);
    for (const auto &message : message_list)
    {
        for (auto index : message.neuron_indexes_)
        {
            if (index < output_size) ++result[index];
        }
    }
    return result;
}
}  // namespace knp::framework::io::output
