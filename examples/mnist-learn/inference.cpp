/**
 * @file inference.cpp
 * @brief Functions for inference.
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

#include "inference.h"

#include <knp/framework/data_processing/image_classification.h>
#include <knp/framework/model.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/model.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>
#include <knp/framework/projection/wta.h>
#include <knp/framework/sonata/network_io.h>

#include <filesystem>
#include <map>
#include <utility>

#include "construct_network.h"
#include "shared_network.h"
#include "time_string.h"

constexpr size_t aggregated_spikes_logging_period = 4e3;

namespace fs = std::filesystem;


std::vector<knp::core::messaging::SpikeMessage> run_mnist_inference(
    const fs::path &path_to_backend, AnnotatedNetwork &described_network,
    knp::framework::data_processing::classification::images::Dataset const &dataset, const fs::path &log_path)
{
    knp::framework::BackendLoader backend_loader;
    knp::framework::Model model(std::move(described_network.network_));

    // Creates arbitrary o_channel_uid identifier for the output channel.
    knp::core::UID o_channel_uid;
    // Passes to the model object the created output channel ID (o_channel_uid)
    // and the population IDs.
    knp::framework::ModelLoader::InputChannelMap channel_map;
    knp::core::UID input_image_channel_uid;
    channel_map.insert(
        {input_image_channel_uid,
         knp::framework::data_processing::classification::images::make_inference_images_spikes_generator(dataset)});

    for (auto i : described_network.data_.output_uids_) model.add_output_channel(o_channel_uid, i);
    for (auto image_proj_uid : described_network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_uid, image_proj_uid);
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));

    // Receives a link to the output channel object (out_channel) from
    // the model executor (model_executor) by the output channel ID (o_channel_uid).
    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.get_backend()->stop_learning();

    knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);

    std::ofstream log_stream;

    // This variable should have the same lifetime as model_executor, or else UB.
    //  cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    std::vector<knp::core::UID> wta_uids;
    {
        std::vector<size_t> wta_borders;
        for (size_t i = 0; i < num_possible_labels; ++i) wta_borders.push_back(neurons_per_column * i);
        wta_uids = knp::framework::projection::add_wta_handlers(
            model_executor, 1, wta_borders, described_network.data_.wta_data_);
    }

    auto all_senders_names = described_network.data_.population_names_;
    for (const auto &uid : wta_uids)
    {
        all_senders_names.insert({uid, "WTA"});
    }

    // All loggers go here
    if (!log_path.empty())
    {
        log_stream.open(log_path / "spikes_inference.csv", std::ofstream::out);
        if (!log_stream.is_open()) std::cout << "Couldn't open log file : " << log_path << std::endl;
    }
    if (log_stream.is_open())
    {
        knp::framework::monitoring::model::add_aggregated_spikes_logger(
            model, all_senders_names, model_executor, spike_accumulator, log_stream, aggregated_spikes_logging_period);
    }

    // Start model.
    std::cout << get_time_string() << ": inference started\n";
    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Inference step: " << step << std::endl;
            return step != dataset.steps_required_for_inference_;
        });
    // Updates the output channel.
    auto spikes = out_channel.update();
    std::sort(
        spikes.begin(), spikes.end(),
        [](const auto &sm1, const auto &sm2) { return sm1.header_.send_time_ < sm2.header_.send_time_; });
    return spikes;
}
