/**
 * @file model.h
 * @brief Functions for network monitoring.
 * @kaspersky_support D. Postnikov
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


/**
 * @brief Namespace for monitoring and logging model behavior during execution. 
 */
namespace knp::framework::monitoring::model
{

/**
 * @brief Function object that receives spike messages and processes their data.
 */
using SpikeProcessor = knp::framework::monitoring::MessageProcessor<knp::core::messaging::SpikeMessage>;


/**
 * @brief Add a logger that outputs spikes in aggregated format.
 * @details The function sets up an observer that aggregates spike counts from specified senders and 
 * writes the aggregated results to the provided output stream at regular intervals.
 * @param model network model.
 * @param senders_names UID-name mapping of senders that will have spike observer attached to them.
 * @param model_executor model executor.
 * @param spike_accumulator buffer for accumulating spike counts between logging intervals.
 * @param log_stream output stream to write aggregated spike counts.
 * @param logging_period interval between logging operations.
 */
KNP_DECLSPEC void add_aggregated_spikes_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &senders_names,
    knp::framework::ModelExecutor &model_executor, std::map<std::string, size_t> &spike_accumulator,
    std::ostream &log_stream, size_t logging_period);


/**
 * @brief Add a logger that outputs synaptic weights from a specific projection.
 * @details The function sets up an observer that captures and logs synaptic weights from the specified 
 * projection at regular intervals. 
 * @param weights_log output stream to write projection weights. 
 * @param model_executor model executor.
 * @param uid UID of the projection to monitor.
 * @param logging_period interval between logging operations.
 */
KNP_DECLSPEC void add_projection_weights_logger(
    std::ostream &weights_log, knp::framework::ModelExecutor &model_executor, knp::core::UID const &uid,
    size_t logging_period);


/**
 * @brief Add a logger that outputs all spike messages with detailed information.
 * @details The function sets up an observer that captures all spike messages from the specified senders and
 * writes detailed spike information to the provided output stream.
 * @param model_executor model executor.
 * @param senders_names UID-name mapping of senders that will have spike observer attached to them.
 * @param log_stream output stream to write spike messages.
 */
KNP_DECLSPEC void add_spikes_logger(
    knp::framework::ModelExecutor &model_executor, const std::map<knp::core::UID, std::string> &senders_names,
    std::ostream &log_stream);


/**
 * @brief Add a logger that outputs model status information.
 * @details The function sets up an observer that monitors spike messages and outputs the count of spike messages
 * received at each logging interval.
 * @param model_executor model executor.
 * @param model model.
 * @param log_stream output stream to write model status information.
 * @param logging_period interval between logging operations.
 */
KNP_DECLSPEC void add_status_logger(
    knp::framework::ModelExecutor &model_executor, const knp::framework::Model &model, std::ostream &log_stream,
    size_t logging_period);

}  //namespace knp::framework::monitoring::model
