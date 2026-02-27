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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "annotated_network.h"
#include "dataset.h"
#include "global_config.h"
#include "models/altai/network_functions.h"
#include "models/blifat/network_functions.h"


/**
 * @brief Build channel map train. Will add input/output channels and add spikes generators.
 * @tparam Neuron Neuron type.
 * @param network Annotated network.
 * @param model Model.
 * @param dataset Dataset.
 * @return Channel map.
 */
template <typename Neuron>
knp::framework::ModelLoader::InputChannelMap build_channel_map_train(
    const AnnotatedNetwork& network, knp::framework::Model& model, const Dataset& dataset)
{
    // Create future channels uids randomly.
    knp::core::UID input_image_channel_raster;
    knp::core::UID input_image_channel_classes;
    knp::core::UID output_channel;

    // Add input channel for each image input projection.
    for (auto image_proj_uid : network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_raster, image_proj_uid);

    // Add input channel for data labels.
    for (auto target_proj_uid : network.data_.projections_from_classes_)
        model.add_input_channel(input_image_channel_classes, target_proj_uid);

    // Add output channel.
    for (auto out_pop : network.data_.output_uids_) model.add_output_channel(output_channel, out_pop);

    // Create and fill a channel map.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=276672
    knp::framework::ModelLoader::InputChannelMap channel_map;
    channel_map.insert({input_image_channel_raster, dataset.make_training_images_spikes_generator()});
    channel_map.insert({input_image_channel_classes, make_training_labels_spikes_generator<Neuron>(dataset)});

    return channel_map;
}


/**
 * @brief Train network.
 * @tparam Neuron Neuron type.
 * @param backend Backend for training.
 * @param network Annotated network.
 * @param model_desc Model description.
 * @param dataset Dataset.
 */
template <typename Neuron>
void train_network(
    const std::shared_ptr<knp::core::Backend>& backend, AnnotatedNetwork& network, const ModelDescription& model_desc,
    const Dataset& dataset)
{
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235849
    knp::framework::Model model(std::move(network.network_));

    knp::framework::ModelLoader::InputChannelMap channel_map = build_channel_map_train<Neuron>(network, model, dataset);

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(model, backend, std::move(channel_map));

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=260375
    knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);

    // Add all spikes observer.
    // All these variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream;
    // cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=301132
    std::vector<knp::core::UID> wta_uids = knp::framework::projection::add_wta_handlers(
        model_executor, wta_winners_amount, network.data_.wta_borders_, network.data_.wta_data_);

    auto pop_names = network.data_.population_names_;

    // Add WTA populations for logging.
    for (auto const& uid : wta_uids) pop_names[uid] = "WTA";

    knp::framework::monitoring::model::add_spikes_logger(model_executor, pop_names, std::cout);

    // All loggers go here
    if (!model_desc.log_path_.empty())
    {
        log_stream.open(model_desc.log_path_ / "spikes_training.csv", std::ofstream::out);
        if (log_stream.is_open())
            knp::framework::monitoring::model::add_aggregated_spikes_logger(
                model, pop_names, model_executor, spike_accumulator, log_stream, aggregated_spikes_logging_period);
        else
            std::cout << "Couldn't open spikes_training.csv at " << model_desc.log_path_ << std::endl;

        weight_stream.open(model_desc.log_path_ / "weights.log", std::ofstream::out);
        if (weight_stream.is_open())
            knp::framework::monitoring::model::add_projection_weights_logger(
                weight_stream, model_executor, network.data_.projections_from_raster_[0],
                projection_weights_logging_period);
        else
            std::cout << "Couldn't open weights.csv at " << model_desc.log_path_ << std::endl;
    }

    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Step: " << step << std::endl;
            return step != dataset.get_steps_amount_for_training();
        });
}


/**
 * @brief Train model.
 * @tparam Neuron Neuron type.
 * @param model_desc Model description.
 * @param dataset Dataset.
 * @param network Annotated network.
 * @param backend_loader Backend loader.
 */
template <typename Neuron>
void train_model(
    const ModelDescription& model_desc, const Dataset& dataset, AnnotatedNetwork& network,
    knp::framework::BackendLoader& backend_loader)
{
    std::shared_ptr<knp::core::Backend> training_backend = backend_loader.load(model_desc.training_backend_path_);
    train_network<Neuron>(training_backend, network, model_desc, dataset);

    prepare_network_for_inference<Neuron>(training_backend, model_desc, network);
}
