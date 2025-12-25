/**
 * @file main.cpp
 * @brief Example of training a MNIST network
 * @kaspersky_support D. Postnikov
 * @date 25.12.2025
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

#include <knp/core/projection.h>
#include <knp/framework/inference_evaluation/classification/processor.h>
#include <knp/framework/sonata/network_io.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "inference.h"
#include "shared.h"
#include "time_string.h"
#include "train.h"

namespace data_processing = knp::framework::data_processing::classification::images;
namespace inference_evaluation = knp::framework::inference_evaluation::classification;

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "You need to provide 2[3] arguments,\n1: path to images raw data\n2: path to images labels\n[3]: "
                     "path to folder for logs"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path images_file_path = argv[1];
    std::filesystem::path labels_file_path = argv[2];

    std::filesystem::path log_path;
    if (4 == argc) log_path = argv[3];

    // Defines path to backend, on which to run a network.
    std::filesystem::path path_to_backend =
        std::filesystem::path(argv[0]).parent_path() / "knp-cpu-single-threaded-backend";

    std::ifstream images_stream(images_file_path, std::ios::binary);
    std::ifstream labels_stream(labels_file_path, std::ios::in);

    data_processing::Dataset dataset;
    dataset.process_labels_and_images(
        images_stream, labels_stream, images_amount_to_train + images_amount_for_inference, classes_amount, input_size,
        steps_per_image, dataset.make_incrementing_image_to_spikes_converter(active_steps, state_increment_factor));
    dataset.split(images_amount_to_train, images_amount_for_inference);

    std::cout << "Processed dataset, training will last " << dataset.get_steps_amount_for_training()
              << " steps, inference " << dataset.get_steps_amount_for_inference() << " steps" << std::endl;

    // Construct network and run training.
    AnnotatedNetwork trained_network = train_mnist_network(path_to_backend, dataset, log_path);

    {  // Quantisize weights.
        for (auto proj = trained_network.network_.begin_projections();
             proj != trained_network.network_.end_projections(); ++proj)
        {
            std::visit(
                [&trained_network](auto&& proj)
                {
                    float max_weight = 0, min_weight = 0;
                    for (auto& synapse : proj)
                    {
                        auto const& params = std::get<knp::core::synapse_data>(synapse);
                        if (max_weight < params.weight_) max_weight = params.weight_;
                        if (min_weight > params.weight_) min_weight = params.weight_;
                    }

                    knp::core::UID post_pop_uid = proj.get_postsynaptic();
                    auto& pop = std::get<knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>>(
                        trained_network.network_.get_population(post_pop_uid));

                    uint16_t max_threshold = 0;
                    for (auto const& neuron : pop)
                        max_threshold = std::max<uint16_t>(
                            max_threshold, neuron.activation_threshold_ + neuron.additional_threshold_);

                    float total_max =
                        std::max({std::abs(max_weight), std::abs(min_weight), std::abs<float>(max_threshold)});
                    float scale = 255.f / total_max;
                    std::cout << scale << std::endl;

                    for (auto& synapse : proj)
                    {
                        auto& params = std::get<knp::core::synapse_data>(synapse);
                        params.weight_ *= scale;
                        params.weight_ = std::round(params.weight_);
                    }

                    for (auto& neuron : pop)
                    {
                        neuron.activation_threshold_ *= scale;
                        neuron.additional_threshold_ *= scale;
                    }
                },
                *proj);
        }
    }

    std::filesystem::create_directory("mnist_network");
    knp::framework::sonata::save_network(trained_network.network_, "mnist_network");

    // Run inference for the same network.
    auto spikes = run_mnist_inference(path_to_backend, trained_network, dataset, log_path);
    std::cout << get_time_string() << ": inference finished  -- output spike count is " << spikes.size() << std::endl;

    // Evaluate results.
    inference_evaluation::InferenceResultsProcessor inference_processor;
    inference_processor.process_inference_results(spikes, dataset);

    inference_processor.write_inference_results_to_stream_as_csv(std::cout);

    return EXIT_SUCCESS;
}
