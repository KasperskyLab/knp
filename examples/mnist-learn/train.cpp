/**
 * @file train.cpp
 * @brief Functions for train network.
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

#include "train.h"

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

constexpr size_t projection_weights_logging_period = 1e5;

constexpr size_t wta_winners_amount = 1;

namespace fs = std::filesystem;

namespace images_classification = knp::framework::data_processing::classification::images;


// Create channel map for training.
auto build_channel_map_train(
    const AnnotatedNetwork &network, knp::framework::Model &model, const images_classification::Dataset &dataset)
{
    // Create future channels uids randomly.
    knp::core::UID input_image_channel_raster;
    knp::core::UID input_image_channel_classes;

    // Add input channel for each image input projection.
    for (auto image_proj_uid : network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_raster, image_proj_uid);

    // Add input channel for data labels.
    for (auto target_proj_uid : network.data_.projections_from_classes_)
        model.add_input_channel(input_image_channel_classes, target_proj_uid);

    // Create and fill a channel map.
    knp::framework::ModelLoader::InputChannelMap channel_map;
    channel_map.insert(
        {input_image_channel_raster, images_classification::make_training_images_spikes_generator(dataset)});
    channel_map.insert({input_image_channel_classes, images_classification::make_training_labels_generator(dataset)});

    return channel_map;
}


knp::framework::Network get_network_for_inference(
    const knp::core::Backend &backend, const std::set<knp::core::UID> &inference_population_uids,
    const std::set<knp::core::UID> &inference_internal_projection)
{
    auto data_ranges = backend.get_network_data();
    knp::framework::Network res_network;

    for (auto &iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, population);
        if (inference_population_uids.find(pop_uid) != inference_population_uids.end())
            res_network.add_population(std::move(population));
    }
    for (auto &iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        auto projection = *iter;
        knp::core::UID proj_uid = std::visit([](const auto &p) { return p.get_uid(); }, projection);
        if (inference_internal_projection.find(proj_uid) != inference_internal_projection.end())
            res_network.add_projection(std::move(projection));
    }
    return res_network;
}


AnnotatedNetwork train_mnist_network(
    const fs::path &path_to_backend, const images_classification::Dataset &dataset, const fs::path &log_path)
{
    AnnotatedNetwork example_network = create_example_network(num_subnetworks);
    std::filesystem::create_directory("mnist_network");
    knp::framework::sonata::save_network(example_network.network_, "mnist_network");
    knp::framework::Model model(std::move(example_network.network_));

    knp::framework::ModelLoader::InputChannelMap channel_map = build_channel_map_train(example_network, model, dataset);

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));

    knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);

    // Add all spikes observer.
    // All these variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream;
    // cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    {
        std::vector<size_t> wta_borders;
        for (size_t i = 0; i < num_possible_labels; ++i) wta_borders.push_back(neurons_per_column * i);
        knp::framework::projection::add_wta_handlers(
            model_executor, wta_winners_amount, wta_borders, example_network.data_.wta_data_);
    }

    // All loggers go here
    if (!log_path.empty())
    {
        log_stream.open(log_path / "spikes_training.csv", std::ofstream::out);
        if (log_stream.is_open())
            knp::framework::monitoring::model::add_aggregated_spikes_logger(
                model, example_network.data_.population_names_, model_executor, spike_accumulator, log_stream,
                aggregated_spikes_logging_period);
        else
            std::cout << "Couldn't open spikes_training.csv at " << log_path << std::endl;

        weight_stream.open(log_path / "weights.log", std::ofstream::out);
        if (weight_stream.is_open())
            knp::framework::monitoring::model::add_projection_weights_logger(
                weight_stream, model_executor, example_network.data_.projections_from_raster_[0],
                projection_weights_logging_period);
        else
            std::cout << "Couldn't open weights.csv at " << log_path << std::endl;
    }

    // Start model.
    std::cout << get_time_string() << ": learning started\n";

    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Step: " << step << std::endl;
            return step != dataset.steps_required_for_training_;
        });

    std::cout << get_time_string() << ": learning finished\n";
    example_network.network_ = get_network_for_inference(
        *model_executor.get_backend(), example_network.data_.inference_population_uids_,
        example_network.data_.inference_internal_projection_);
    return example_network;
}
