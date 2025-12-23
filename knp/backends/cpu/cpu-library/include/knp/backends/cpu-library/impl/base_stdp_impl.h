/**
 * @file base_stdp_impl.h
 * @brief Common templates used for different STDP variants.
 * @kaspersky_support A. Vartenkov
 * @date 04.11.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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
#include <knp/core/projection.h>

#include <unordered_map>
#include <vector>

/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
template <class DeltaLikeSynapse>
struct WeightUpdateSTDP
{
    static void init_projection_part(
        const knp::core::Projection<DeltaLikeSynapse> &projection,
        const std::vector<core::messaging::SpikeMessage> &messages, uint64_t part_begin, uint64_t part_end,
        uint64_t step)
    {
    }

    static void init_synapse(knp::synapse_traits::synapse_parameters<DeltaLikeSynapse> &params, uint64_t step) {}

    static void modify_weights_part(
        const knp::core::Projection<DeltaLikeSynapse> &projection, uint64_t part_begin, uint64_t part_end)
    {
    }

    static void init_projection(
        const knp::core::Projection<DeltaLikeSynapse> &projection,
        const std::vector<core::messaging::SpikeMessage> &messages, uint64_t step)
    {
    }


    static void modify_weights(const knp::core::Projection<DeltaLikeSynapse> &projection) {}
};

}  // namespace knp::backends::cpu
