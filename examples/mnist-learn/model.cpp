/**
 * @file model.cpp
 * @brief Everything you need to work with model.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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


#include "model.h"

#include <knp/framework/model.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/model.h>
#include <knp/framework/projection/wta.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "models/altai/construct_network.h"
#include "models/blifat/construct_network.h"
#include "settings.h"


AnnotatedNetwork construct_network(const ModelDescription& model_desc)
{
    switch (model_desc.type_)
    {
        case SupportedModelType::BLIFAT:
        {
            return construct_network_blifat(model_desc);
        }
        case SupportedModelType::AltAI:
        {
            return construct_network_altai(model_desc);
        }
        default:
            throw std::runtime_error("Unknown model type.");
    }
}


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
    // channel_map.insert({input_image_channel_classes, dataset.make_training_labels_generator()});
    channel_map.insert(
        {input_image_channel_classes, [&dataset](knp::core::Step step)
         {
             knp::core::messaging::SpikeData message;

             knp::core::Step local_step = step % steps_per_image;
             if (local_step == 11)
                 message.push_back(dataset.get_data_for_training().first[step / steps_per_image].first);
             return message;
         }});


    return channel_map;
}

void train_network(AnnotatedNetwork& network, const ModelDescription& model_desc, const Dataset& dataset)
{
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235849
    knp::framework::Model model(std::move(network.network_));

    knp::framework::ModelLoader::InputChannelMap channel_map = build_channel_map_train(network, model, dataset);

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    knp::framework::BackendLoader backend_loader;
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(model_desc.backend_path_), std::move(channel_map));

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
        wta_uids = knp::framework::projection::add_wta_handlers(
            model_executor, wta_winners_amount, network.data_.wta_borders_, network.data_.wta_data_);
    }

    auto pop_names = network.data_.population_names_;

    /*
    // Change names for WTA populations.
    {
        for (auto pop = pop_names.begin(); pop != pop_names.end(); ++pop)
            if (pop->second == "INPUT") pop->second = "INPUT[NO WTA]";
        for (auto const& uid : wta_uids) pop_names[uid] = "INPUT[WTA]";
    }
    */

    // knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);
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

    // TOOD this is temporary solution. backend should be loaded only once, not twice, so this line can be moved to
    // main.
    strip_network_for_inference(*model_executor.get_backend(), network, model_desc);
}


void strip_network_for_inference(
    const knp::core::Backend& backend, AnnotatedNetwork& network, const ModelDescription& model_desc)
{
    auto data_ranges = backend.get_network_data();
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
    network.network_ = knp::framework::Network();

    for (auto& iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto& p) { return p.get_uid(); }, population);
        if (network.data_.inference_population_uids_.find(pop_uid) != network.data_.inference_population_uids_.end())
            network.network_.add_population(std::move(population));
    }
    for (auto& iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235844
        auto projection = *iter;
        knp::core::UID proj_uid = std::visit([](const auto& p) { return p.get_uid(); }, projection);
        if (network.data_.inference_internal_projection_.find(proj_uid) !=
            network.data_.inference_internal_projection_.end())
            network.network_.add_projection(std::move(projection));
    }
}


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

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=260375
    knp::framework::monitoring::model::add_status_logger(model_executor, model, std::cout, 1);

    std::ofstream log_stream;

    // This variable should have the same lifetime as model_executor, or else UB.
    //  cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=301132
    std::vector<knp::core::UID> wta_uids;
    {
        wta_uids = knp::framework::projection::add_wta_handlers(
            model_executor, wta_winners_amount, network.data_.wta_borders_, network.data_.wta_data_);
    }

    auto pop_names = network.data_.population_names_;

    /*
    // Change names for WTA populations.
    {
        for (auto pop = pop_names.begin(); pop != pop_names.end(); ++pop)
            if (pop->second == "INPUT") pop->second = "INPUT[NO WTA]";
        for (auto const& uid : wta_uids) pop_names[uid] = "INPUT[WTA]";
    }
    */

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
