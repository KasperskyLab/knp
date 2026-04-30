/**
 * @file inference.h
 * @brief Functions for inference.
 * @kaspersky_support A. Vartenkov
 * @date 24.03.2025
 * @license Apache 2.0
 * @copyright © 2025-2026 AO Kaspersky Lab
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

#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/model.h>
#include <knp/framework/projection/wta.h>
#include <knp/framework/tags/name.h>

#include <map>
#include <memory>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "annotated_network.h"
#include "dataset.h"
#include "global_config.h"


/**
 * @brief Run inference on a neural network and record spike activity.
 * 
 * @details This function executes the inference process on a trained neural network, processing input data through the network 
 * and recording spike messages for analysis. It configures the network with appropriate input and output channels, sets up 
 * WTA mechanisms, initializes logging for spike monitoring, and executes the simulation.
 * 
 * @tparam Neuron neuron type template parameter for neuron model specification.
 * 
 * @param backend shared pointer to the computational backend for execution.
 * @param network annotated network structure containing the network and its annotations.
 * @param model_desc model description containing configuration parameters and paths.
 * @param dataset dataset with inference data.
 * @return vector of spike messages recorded during inference execution.
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> infer_network(
    const std::shared_ptr<knp::core::Backend>& backend, AnnotatedNetwork& network, const ModelDescription& model_desc,
    const Dataset& dataset)
{
    // Store population names for logging and debugging purposes.
    std::map<knp::core::UID, std::string> pop_names;
    for (const auto& pop : network.network_.get_populations())
        std::visit(
            [&pop_names](const auto& pop) { pop_names[pop.get_uid()] = knp::framework::tags::get_name(pop); }, pop);

    // Create model from annotated network structure.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235849
    knp::framework::Model model(std::move(network.network_));

    // Create output channel UID (o_channel_uid) for network outputs.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243539
    knp::core::UID o_channel_uid;

    // Pass the created output channel ID (o_channel_uid) and the population IDs to the model object.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=276672
    knp::framework::ModelLoader::InputChannelMap channel_map;

    // Create input channel UID for image spikes.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=244944
    knp::core::UID input_image_channel_uid;
    channel_map.insert({input_image_channel_uid, dataset.make_inference_images_spikes_generator()});

    // Connect output populations to the output channel.
    for (auto i : network.data_.output_uids_) model.add_output_channel(o_channel_uid, i);

    // Connect rasterized image projections to the input channel.
    for (auto image_proj_uid : network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_uid, image_proj_uid);
    
    // Initialize model executor with backend and channel mappings.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(model, backend, std::move(channel_map));

    // Get reference to output channel for retrieving results.
    auto& out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.get_backend()->stop_learning();

    // Initialize logging streams.
    std::ofstream log_stream;
    std::ofstream raw_spikes_stream;

    // Spike accumulator for aggregated logging (must live as long as model_executor)
    //  cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    // Configure WTA (Winner-Take-All) handlers for competitive learning.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=301132
    std::vector<knp::core::UID> wta_uids = knp::framework::projection::add_wta_handlers(
        model_executor, wta_winners_amount, network.data_.wta_borders_, network.data_.wta_data_);

    // Register WTA populations for logging.
    for (auto const& uid : wta_uids) pop_names[uid] = "WTA";

    // Add real-time spike logger for console output.
    knp::framework::monitoring::model::add_spikes_logger(model_executor, pop_names, std::cout);

    // Setup detailed spike logging if log path is specified.
    if (!model_desc.log_path_.empty())
    {
        std::filesystem::create_directories(model_desc.log_path_);
        raw_spikes_stream.open(model_desc.log_path_ / "spikes_inference_raw.csv", std::ofstream::out);
        if (!raw_spikes_stream.is_open())
        {
            std::cout << "Couldn't open raw inference spikes log file : " << model_desc.log_path_ << std::endl;
        }
        else
        {
            // Write CSV header for raw spike data.
            raw_spikes_stream << "send_time,sender_name,sender_uid,neuron_index" << std::endl;

            // Collect all sender UIDs for observer registration.
            std::vector<knp::core::UID> all_senders_uids(pop_names.size());
            std::transform(
                pop_names.begin(), pop_names.end(), all_senders_uids.begin(),
                [](const auto& sender) -> knp::core::UID { return sender.first; });

            // Register observer to capture and log raw spike messages.
            model_executor.add_observer<knp::core::messaging::SpikeMessage>(
                [&raw_spikes_stream, &pop_names](const std::vector<knp::core::messaging::SpikeMessage>& messages)
                {
                    for (const auto& message : messages)
                    {
                        const auto name_iter = pop_names.find(message.header_.sender_uid_);
                        const std::string sender_name =
                            name_iter == pop_names.end() ? "UNKNOWN" : name_iter->second;

                        // Log each neuron index from the spike message.
                        for (const auto neuron_index : message.neuron_indexes_)
                        {
                            raw_spikes_stream << message.header_.send_time_ << "," << sender_name << ","
                                              << message.header_.sender_uid_ << "," << neuron_index << std::endl;
                        }
                    }
                },
                all_senders_uids);
        }
    }

    // Setup aggregated spike logging if enabled.
    if (!model_desc.log_path_.empty())
    {
        log_stream.open(model_desc.log_path_ / "spikes_inference.csv", std::ofstream::out);
        if (!log_stream.is_open()) std::cout << "Couldn't open log file : " << model_desc.log_path_ << std::endl;
    }

    // Enable aggregated spikes logging if file is successfully opened.
    if (log_stream.is_open())
    {
        knp::framework::monitoring::model::add_aggregated_spikes_logger(
            model, pop_names, model_executor, spike_accumulator, log_stream, aggregated_spikes_logging_period);
    }

    // Execute the inference simulation with progress reporting.
    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Inference step: " << step << std::endl;
            return step != dataset.get_steps_amount_for_inference();
        });
    
    // Retrieve final spike results from output channel.
    auto spikes = out_channel.update();

    // Sort spikes by send time for consistent ordering.
    std::sort(
        spikes.begin(), spikes.end(),
        [](const auto& sm1, const auto& sm2) { return sm1.header_.send_time_ < sm2.header_.send_time_; });
    return spikes;
}


/**
 * @brief Run inference on a model and record spike activity.
 * 
 * @details This function provides a high-level interface for running inference on a trained model, handling backend loading and 
 * delegation to the lower-level inference function.
 * 
 * @tparam Neuron neuron type template parameter for neuron model specification.
 * 
 * @param model_desc model description containing configuration parameters and paths.
 * @param dataset dataset with inference data.
 * @param network annotated network structure containing the network and its annotations.
 * @param backend_loader backend loader.
 * @return vector of spike messages recorded during inference execution.
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> infer_model(
    const ModelDescription& model_desc, const Dataset& dataset, AnnotatedNetwork& network,
    knp::framework::BackendLoader& backend_loader)
{
    std::shared_ptr<knp::core::Backend> inference_backend = backend_loader.load(model_desc.inference_backend_path_);
    return infer_network<Neuron>(inference_backend, network, model_desc, dataset);
}
