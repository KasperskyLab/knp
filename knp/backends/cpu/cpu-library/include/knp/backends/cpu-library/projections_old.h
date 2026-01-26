/**
 * @file projections_old.h
 * @kaspersky_support Postnikov D.
 * @date 26.01.2026
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

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/projection.h>

#include "projections.h"


/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{

/**
 * @brief Make one execution step for a projection of delta synapses.
 * @tparam DeltaLikeSynapseType type of a synapse that requires synapse weight and delay as parameters.
 * @param proj projection to update.
 * @param endpoint message endpoint used for message exchange.
 * @param future_messages message queue to process via endpoint.
 * @param step_n execution step.
 */
template <class DeltaLikeSynapseType>
void calculate_delta_synapse_projection(
    knp::core::Projection<DeltaLikeSynapseType> &proj, knp::core::MessageEndpoint &endpoint,
    projections::MessageQueue &future_messages, size_t step_n)
{
    projections::calculate_projection(proj, endpoint, future_messages, step_n);
}

}  // namespace knp::backends::cpu
