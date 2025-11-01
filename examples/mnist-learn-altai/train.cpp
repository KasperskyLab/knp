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
    channel_map.insert({input_image_channel_raster, dataset.make_training_images_spikes_generator()});
    channel_map.insert({input_image_channel_classes, dataset.make_training_labels_generator()});

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
    /*
     * 1 network
0,21,21,0,0,229,1,1,0.084,1,1
1,26,19,6,1,224,0.95,0.95,0.1,0.972,0.95
2,23,8,0,15,227,0.347826,0.347826,0.032,0.94,0.347826
3,23,6,0,17,227,0.26087,0.26087,0.024,0.932,0.26087
4,23,13,1,9,227,0.590909,0.590909,0.056,0.96,0.590909
5,29,0,0,29,221,0,0,0,0.884,0
6,24,13,0,11,226,0.541667,0.541667,0.052,0.956,0.541667
7,30,23,1,6,220,0.793103,0.793103,0.096,0.972,0.793104
8,22,4,0,18,228,0.181818,0.181818,0.016,0.928,0.181818
9,29,1,0,28,221,0.0344828,0.0344828,0.004,0.888,0.0344828
*average 0.46

     * 3 networks
CLASS,TOTAL_VOTES,TRUE_POSITIVES,FALSE_NEGATIVES,FALSE_POSITIVES,TRUE_NEGA
TIVES,PRECISION,RECALL,PREVALENCE,ACCURACY,F_SCORE
0,21,21,0,0,229,1,1,0.084,1,1
1,26,12,13,1,224,0.923077,0.923077,0.1,0.944,0.923077
2,23,15,0,8,227,0.652174,0.652174,0.06,0.968,0.652174
3,23,10,0,13,227,0.434783,0.434783,0.04,0.948,0.434783
4,23,18,1,4,227,0.818182,0.818182,0.076,0.98,0.818182
5,29,0,1,28,221,0,0,0.004,0.884,0
6,24,13,0,11,226,0.541667,0.541667,0.052,0.956,0.541667
7,30,23,2,5,220,0.821429,0.821429,0.1,0.972,0.821429
8,22,6,0,16,228,0.272727,0.272727,0.024,0.936,0.272727
9,29,3,1,25,221,0.107143,0.107143,0.016,0.896,0.107143
*average 0.62

* 6 networks
CLASS,TOTAL_VOTES,TRUE_POSITIVES,FALSE_NEGATIVES,FALSE_POSITIVES,TRUE_NEGA
TIVES,PRECISION,RECALL,PREVALENCE,ACCURACY,F_SCORE
0,21,21,0,0,229,1,1,0.084,1,1
1,26,21,4,1,224,0.954545,0.954545,0.1,0.98,0.954545
2,23,15,0,8,227,0.652174,0.652174,0.06,0.968,0.652174
3,23,14,0,9,227,0.608696,0.608696,0.056,0.964,0.608696
4,23,18,1,4,227,0.818182,0.818182,0.076,0.98,0.818182
5,29,0,0,29,221,0,0,0,0.884,0
6,24,15,0,9,226,0.625,0.625,0.06,0.964,0.625
7,30,26,1,3,220,0.896552,0.896552,0.108,0.984,0.896552
8,22,13,0,9,228,0.590909,0.590909,0.052,0.964,0.590909
9,29,4,0,25,221,0.137931,0.137931,0.016,0.9,0.137931
*average 0.65

     */
    AnnotatedNetwork example_network = create_example_network(1);  // num_subnetworks);
    std::filesystem::create_directory("mnist_network");
    knp::framework::sonata::save_network(example_network.network_, "mnist_network");
    knp::framework::Model model(std::move(example_network.network_));

    knp::framework::ModelLoader::InputChannelMap channel_map = build_channel_map_train(example_network, model, dataset);

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));

    /*
     как михаил подает картинку?

    что за комент про reset potential?
    наш gating синапс просто не дает выдавать спайки

    у нас цифры перемешанны,
    у таргета задержка 11, и тогда он блокирует слудющие 10 тактов у следующего числа,
    если два одинаковых числа идут подряд, то может быть проблемой?

    спросить логи у михаила
     */

    // Add all spikes observer.
    // All these variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream;
    // cppcheck-suppress variableScope
    std::map<std::string, size_t> spike_accumulator;

    std::vector<knp::core::UID> wta_uids;
    {
        std::vector<size_t> wta_borders;
        for (size_t i = 0; i < num_possible_labels; ++i) wta_borders.push_back(3 * (i + 1));
        wta_uids = knp::framework::projection::add_wta_handlers(
            model_executor, wta_winners_amount, wta_borders, example_network.data_.wta_data_);
    }

    auto pop_names = example_network.data_.population_names_;

    // Change uid for WTA population.
    {
        for (auto pop = pop_names.begin(); pop != pop_names.end();)
            if (pop->second == "WTA")
                pop = pop_names.erase(pop);
            else
                ++pop;
        for (auto const &uid : wta_uids) pop_names[uid] = "WTA";
    }

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
            return step != dataset.get_steps_required_for_training();
        });

    std::cout << get_time_string() << ": learning finished\n";
    example_network.network_ = get_network_for_inference(
        *model_executor.get_backend(), example_network.data_.inference_population_uids_,
        example_network.data_.inference_internal_projection_);
    return example_network;
}
