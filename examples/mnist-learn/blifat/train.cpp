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
#include "shared.h"
#include "time_string.h"

namespace fs = std::filesystem;

namespace images_classification = knp::framework::data_processing::classification::images;


// Create channel map for training.
auto build_channel_map_train(
    const AnnotatedNetwork &network, knp::framework::Model &model, const images_classification::Dataset &dataset)
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
    // channel_map.insert({input_image_channel_classes, dataset.make_training_labels_generator()});
    channel_map.insert(
        {input_image_channel_classes, [&dataset](knp::core::Step step)
         {
             knp::core::messaging::SpikeData message;

             knp::core::Step local_step = step % 15;
             if (local_step == 11) message.push_back(dataset.get_data_for_training()[step / 15].first);
             std::cout << "image label: step " << step << '\n'
                       << (message.size() ? static_cast<int>(*message.rbegin()) : -1) << std::endl;
             return message;
         }});


    return channel_map;
}


knp::framework::Network get_network_for_inference(
    const knp::core::Backend &backend, const std::set<knp::core::UID> &inference_population_uids,
    const std::set<knp::core::UID> &inference_internal_projection)
{
    auto data_ranges = backend.get_network_data();
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
    knp::framework::Network res_network;

    for (auto &iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, population);
        if (inference_population_uids.find(pop_uid) != inference_population_uids.end())
            res_network.add_population(std::move(population));
    }
    for (auto &iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235844
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
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235849
    knp::framework::Model model(std::move(example_network.network_));

    knp::framework::ModelLoader::InputChannelMap channel_map = build_channel_map_train(example_network, model, dataset);

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    knp::framework::BackendLoader backend_loader;
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=260375
    knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);

    // Add all spikes observer.
    // All these variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream;
    // cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    std::vector<knp::core::UID> wta_uids;
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=301132
        std::vector<size_t> wta_borders;
        for (size_t i = 0; i < classes_amount; ++i) wta_borders.push_back(neurons_per_column * (i + 1));
        wta_uids = knp::framework::projection::add_wta_handlers(
            model_executor, wta_winners_amount, wta_borders, example_network.data_.wta_data_);
    }

    auto pop_names = example_network.data_.population_names_;

    // Change names for WTA populations.
    {
        for (auto pop = pop_names.begin(); pop != pop_names.end(); ++pop)
            if (pop->second == "INPUT") pop->second = "INPUT[NO WTA]";
        for (auto const &uid : wta_uids) pop_names[uid] = "INPUT[WTA]";
    }

    // knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);
    knp::framework::monitoring::model::add_spikes_logger(model_executor, pop_names, std::cout);

    // All loggers go here
    if (!log_path.empty())
    {
        log_stream.open(log_path / "spikes_training.csv", std::ofstream::out);
        if (log_stream.is_open())
            knp::framework::monitoring::model::add_aggregated_spikes_logger(
                model, pop_names, model_executor, spike_accumulator, log_stream, aggregated_spikes_logging_period);
        else
            std::cout << "Couldn't open spikes_training.csv at " << log_path << std::endl;

        weight_stream.open(log_path / "weights.log", std::ofstream::out);
        if (weight_stream.is_open())
            knp::framework::monitoring::model::add_projection_weights_logger(
                weight_stream, model_executor, example_network.data_.projections_from_raster_[0],
                20000);  // projection_weights_logging_period);
        else
            std::cout << "Couldn't open weights.csv at " << log_path << std::endl;
    }

    // Start model.
    std::cout << get_time_string() << ": learning started\n";

    model_executor.start(
        [&dataset](size_t step)
        {
            if (step % 20 == 0) std::cout << "Step: " << step << std::endl;
            return step != dataset.get_steps_amount_for_training();
        });

    std::cout << get_time_string() << ": learning finished\n";
    example_network.network_ = get_network_for_inference(
        *model_executor.get_backend(), example_network.data_.inference_population_uids_,
        example_network.data_.inference_internal_projection_);
    return example_network;
}
