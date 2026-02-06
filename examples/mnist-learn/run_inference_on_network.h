/**
 * @file run_inference_on_network.h
 * @brief Function for running inference on network.
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
#include <string>
#include <utility>
#include <vector>

#include "annotated_network.h"
#include "dataset.h"
#include "settings.h"


/**
 * @brief Run inference on network, and record spikes.
 * @tparam Neuron Neuron type.
 * @param network Annotated network.
 * @param model_desc Model description.
 * @param dataset Dataset.
 * @return Recorded spikes.
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> run_inference_on_network(
    AnnotatedNetwork& network, const ModelDescription& model_desc, const Dataset& dataset)
{
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    knp::framework::BackendLoader backend_loader;
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235849
    knp::framework::Model model(std::move(network.network_));

    // Creates arbitrary o_channel_uid identifier for the output channel.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243539
    knp::core::UID o_channel_uid;
    // Passes the created output channel ID (o_channel_uid) and the population IDs to the model object.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=276672
    knp::framework::ModelLoader::InputChannelMap channel_map;
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=244944
    knp::core::UID input_image_channel_uid;
    channel_map.insert({input_image_channel_uid, dataset.make_inference_images_spikes_generator()});

    for (auto i : network.data_.output_uids_) model.add_output_channel(o_channel_uid, i);
    for (auto image_proj_uid : network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_uid, image_proj_uid);
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(model_desc.backend_path_), std::move(channel_map));

    // Receives a link to the output channel object (out_channel) from
    // the model executor (model_executor) by the output channel ID (o_channel_uid).
    auto& out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.get_backend()->stop_learning();

    std::ofstream log_stream;

    // This variable should have the same lifetime as model_executor, or else UB.
    //  cppcheck-suppress variableScope
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
        log_stream.open(model_desc.log_path_ / "spikes_inference.csv", std::ofstream::out);
        if (!log_stream.is_open()) std::cout << "Couldn't open log file : " << model_desc.log_path_ << std::endl;
    }
    if (log_stream.is_open())
    {
        knp::framework::monitoring::model::add_aggregated_spikes_logger(
            model, pop_names, model_executor, spike_accumulator, log_stream, aggregated_spikes_logging_period);
    }

    // Start model.
    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Inference step: " << step << std::endl;
            return step != dataset.get_steps_amount_for_inference();
        });
    // Updates the output channel.
    auto spikes = out_channel.update();
    std::sort(
        spikes.begin(), spikes.end(),
        [](const auto& sm1, const auto& sm2) { return sm1.header_.send_time_ < sm2.header_.send_time_; });
    return spikes;
}
