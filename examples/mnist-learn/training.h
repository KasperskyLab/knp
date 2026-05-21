/**
 * @file training.h
 * @brief Functions for training.
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
#include "models/altai/network_functions.h"
#include "models/blifat/network_functions.h"


/**
 * @brief Build input channel map for training operations.
 * 
 * @details This function configures the input and output channels required for training by setting up spike generators 
 * for image data and labels, and establishing the necessary connections between network components.
 * 
 * @tparam Neuron neuron type for neuron model specification.
 * 
 * @param network annotated network structure containing network and annotations.
 * @param model model.
 * @param dataset dataset containing training data and configuration.
 * 
 * @return channel map containing configured input channels for training.
 */
template <typename Neuron>
knp::framework::ModelLoader::InputChannelMap build_channel_map_train(
    const AnnotatedNetwork& network, knp::framework::Model& model, const Dataset& dataset)
{
    // Create random unique identifiers for input and output channels.
    knp::core::UID input_image_channel_raster;
    knp::core::UID input_image_channel_classes;
    knp::core::UID output_channel;

    // Connect rasterized image projections to input channel for image data.
    for (auto image_proj_uid : network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_raster, image_proj_uid);

    // Connect class label projections to input channel for target labels.
    for (auto target_proj_uid : network.data_.projections_from_classes_)
        model.add_input_channel(input_image_channel_classes, target_proj_uid);

    // Connect output populations to output channel for results.
    for (auto out_pop : network.data_.output_uids_) model.add_output_channel(output_channel, out_pop);

    // Create and populate channel map for model execution.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=276672
    knp::framework::ModelLoader::InputChannelMap channel_map;
    channel_map.insert({input_image_channel_raster, dataset.make_training_images_spikes_generator()});
    channel_map.insert({input_image_channel_classes, make_training_labels_spikes_generator<Neuron>(dataset)});

    return channel_map;
}


/**
 * @brief Train neural network on the provided dataset.
 * 
 * @details This function executes the complete training process for a neural network, including model initialization, 
 * training execution, and comprehensive monitoring. It handles the configuration of input and output channels, 
 * logging, and progress tracking.
 * 
 * @tparam Neuron neuron type for neuron model specification.
 * 
 * @param backend shared pointer to the computational backend for training execution.
 * @param network annotated network structure containing the network and its annotations.
 * @param model_desc model description containing configuration parameters and paths.
 * @param dataset dataset containing training data and timing specifications.
 */
template <typename Neuron>
void train_network(
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

    // Build channel map for training operations with proper input and output connections.
    knp::framework::ModelLoader::InputChannelMap channel_map = build_channel_map_train<Neuron>(network, model, dataset);

    // Initialize model executor with backend and channel map for execution.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(model, backend, std::move(channel_map));

    // Add status logger to monitor training progress and system state
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=260375
    knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);

    // Initialize logging streams and accumulators for detailed monitoring.
    // All these variables should have the same lifetime as model_executor.
    std::ofstream log_stream, weight_stream;
    // cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    // Configure WTA (Winner-Take-All) handlers for competitive learning mechanisms.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=301132
    std::vector<knp::core::UID> wta_uids = knp::framework::projection::add_wta_handlers(
        model_executor, wta_winners_amount, network.data_.wta_borders_, network.data_.wta_data_);
    
    // Register WTA populations for logging to track competitive behavior.
    for (auto const& uid : wta_uids) pop_names[uid] = "WTA";

    // Add real-time spike logger for console output during training.
    knp::framework::monitoring::model::add_spikes_logger(model_executor, pop_names, std::cout);

    // Setup detailed logging infrastructure if enabled in configuration.
    if (!model_desc.log_path_.empty())
    {
        // Create logging directory structure if it doesn't exist.
        std::filesystem::create_directories(model_desc.log_path_);

        // Setup aggregated spikes logging for detailed analysis.
        log_stream.open(model_desc.log_path_ / "spikes_training.csv", std::ofstream::out);
        if (log_stream.is_open())
            knp::framework::monitoring::model::add_aggregated_spikes_logger(
                model, pop_names, model_executor, spike_accumulator, log_stream, aggregated_spikes_logging_period);
        else
            std::cout << "Couldn't open spikes_training.csv at " << model_desc.log_path_ << std::endl;

        // Setup weight logging for tracking synaptic changes during training.
        weight_stream.open(model_desc.log_path_ / "weights.log", std::ofstream::out);
        if (weight_stream.is_open())
            knp::framework::monitoring::model::add_projection_weights_logger(
                weight_stream, model_executor, network.data_.projections_from_raster_[0],
                projection_weights_logging_period);
        else
            std::cout << "Couldn't open weights.csv at " << model_desc.log_path_ << std::endl;
    }

    // Execute training simulation with progress reporting.
    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Step: " << step << std::endl;
            return step != dataset.get_steps_amount_for_training();
        });
}


/**
 * @brief Train neural network model and prepare for inference.
 * 
 * @details This function orchestrates the complete training workflow, including backend loading, network training, and 
 * post-training preparation for inference operations. It provides a high-level interface for executing the full training 
 * pipeline.
 * 
 * @tparam Neuron neuron type for neuron model specification.
 * 
 * @param model_desc model description containing configuration parameters and paths.
 * @param dataset dataset containing training data and timing specifications.
 * @param network annotated network structure containing the network and its annotations.
 * @param backend_loader backend loader for loading the training backend.
 */
template <typename Neuron>
void train_model(
    const ModelDescription& model_desc, const Dataset& dataset, AnnotatedNetwork& network,
    knp::framework::BackendLoader& backend_loader)
{
    // Load training backend for computational execution.
    std::shared_ptr<knp::core::Backend> training_backend = backend_loader.load(model_desc.training_backend_path_);

    // Execute the complete training process with the loaded backend.
    train_network<Neuron>(training_backend, network, model_desc, dataset);

    // Prepare the trained network for inference operations.
    prepare_network_for_inference<Neuron>(training_backend, model_desc, network);
}
