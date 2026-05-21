/**
 * @file projections_old.h
 * @brief Legacy interface for working with projections for backend.
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
 * @brief Namespace for CPU backend functions.
 */
namespace knp::backends::cpu
{

/**
 * @brief Execute one simulation step for a projection of delta synapses.
 *
 * @tparam DeltaLikeSynapseType type of a synapse that possesses Delta‑like parameters.
 *
 * @param proj projection to calculate.
 * @param endpoint message endpoint used for loading and sending messages.
 * @param future_messages queue that stores messages to be processed in future steps.
 * @param step_n current simulation step number.
 *
 * @details This function is a thin wrapper that forwards all arguments to 
 * @ref projections::calculate_projection, which performs the actual computation for the delta
 * synapse projection.
 */
template <class DeltaLikeSynapseType>
void calculate_delta_synapse_projection(
    knp::core::Projection<DeltaLikeSynapseType> &proj, knp::core::MessageEndpoint &endpoint,
    projections::MessageQueue &future_messages, size_t step_n)
{
    projections::calculate_projection(proj, endpoint, future_messages, step_n);
}

}  // namespace knp::backends::cpu
