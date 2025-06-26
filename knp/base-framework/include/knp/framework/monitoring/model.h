/**
 * @file model.h
 * @brief Functions for network construction.
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


namespace knp::framework::monitoring::model
{

/**
 * @brief Function object that receives messages and processes their data somehow.
 */
using SpikeProcessor = knp::framework::monitoring::MessageProcessor<knp::core::messaging::SpikeMessage>;


/**
 * @brief add a logger that outputs spikes in aggregated format
 * @param model network model
 * @param senders_names uid-name of senders that will have spikes observer attached to them
 * @param model_executor model executor
 * @param spike_accumulator will save spikes between logging, acts as a buffer
 * @param log_stream output stream
 * @param logging_period logging period, will output to log_stream each logging_period steps
 */
KNP_DECLSPEC void add_aggregated_spikes_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &senders_names,
    knp::framework::ModelExecutor &model_executor, std::map<std::string, size_t> &spike_accumulator,
    std::ostream &log_stream, size_t logging_period);


/**
 * @brief add a logger that outputs weights from projections
 * @param weights_log output stream
 * @param model_executor model executor
 * @param uid projection uid
 * @param logging_period logging period, will output to weights_log every logging_period steps
 */
KNP_DECLSPEC void add_projection_weights_logger(
    std::ostream &weights_log, knp::framework::ModelExecutor &model_executor, knp::core::UID const &uid,
    size_t logging_period);


/**
 * @brief add a logger that outputs all spikes
 * @param model_executor model executor
 * @param senders_names uid-name of senders that will have spikes observer attached to them
 * @param log_stream output stream
 */
KNP_DECLSPEC void add_spikes_logger(
    knp::framework::ModelExecutor &model_executor, const std::map<knp::core::UID, std::string> &senders_names,
    std::ostream &log_stream);


/**
 * @brief add a logger that will output status of model
 * @param model_executor model executor
 * @param model model
 * @param log_stream output stream
 * @param logging_period logging period, will output status to log_stream every logging_period steps
 */
KNP_DECLSPEC void add_status_logger(
    knp::framework::ModelExecutor &model_executor, const knp::framework::Model &model, std::ostream &log_stream,
    size_t logging_period);

}  //namespace knp::framework::monitoring::model
