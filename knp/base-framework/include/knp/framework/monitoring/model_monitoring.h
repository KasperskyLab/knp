/**
 * @file model_monitoring.h
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 24.03.2025
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
#include <knp/core/impexp.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/uid.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/observer.h>

#include <map>
#include <string>
#include <vector>

namespace knp::framework::monitoring::model_monitoring
{


/**
 * @brief Function object that receives messages and processes their data somehow.
 */
using SpikeProcessor = knp::framework::monitoring::MessageProcessor<knp::core::messaging::SpikeMessage>;


/**
 * @brief Add a logger that calculates all spikes from a projection and writes them to a csv file.
 * @param model network model.
 * @param all_senders_names sender UID-name correspondence.
 * @param model_executor model executor.
 * @param current_index parameter to save integer state between runs.
 * @param spike_accumulator parameter to save accumulated spikes between runs.
 * @param log_stream output file stream.
 * @param aggregation_period aggregation period.
 */
void KNP_DECLSPEC add_aggregated_spikes_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &senders_names,
    knp::framework::ModelExecutor &model_executor, size_t &current_index,
    std::map<std::string, size_t> &spike_accumulator, std::ostream &log_stream, size_t logging_period);


void KNP_DECLSPEC add_projection_weights_logger(
    std::ostream &weights_log, knp::framework::ModelExecutor &model_executor, knp::core::UID const &uid,
    size_t logging_period);


void KNP_DECLSPEC add_spikes_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &senders_names,
    knp::framework::ModelExecutor &model_executor, std::ostream &log_stream);


}  //namespace knp::framework::monitoring::model_monitoring
