/**
 * @file main.cpp
 * @brief Example of training a MNIST network
 * @kaspersky_support A. Vartenkov
 * @date 30.08.2024
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

#include <knp/framework/inference_evaluation/classification/processor.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "inference.h"
#include "shared_network.h"
#include "time_string.h"
#include "train.h"

constexpr size_t active_steps = 10;
constexpr size_t steps_per_image = 20;
constexpr float state_increment_factor = 1.f / 255;
constexpr size_t images_amount_to_train = 10000;
constexpr float dataset_split = 0.8;
constexpr size_t classes_amount = 10;

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
        std::filesystem::path(argv[0]).parent_path() / "knp-cpu-multi-threaded-backend";

    std::ifstream images_stream(images_file_path, std::ios::binary);
    std::ifstream labels_stream(labels_file_path, std::ios::in);

    data_processing::Dataset dataset;
    dataset.process_labels_and_images(
        images_stream, labels_stream, images_amount_to_train, classes_amount, input_size, steps_per_image,
        dataset.make_incrementing_image_to_spikes_converter(active_steps, state_increment_factor));
    dataset.split(dataset_split);

    std::cout << "Processed dataset, training will last " << dataset.get_steps_required_for_training()
              << " steps, inference " << dataset.get_steps_required_for_inference() << " steps" << std::endl;

    // Construct network and run training.
    AnnotatedNetwork trained_network = train_mnist_network(path_to_backend, dataset, log_path);

    // Run inference for the same network.
    auto spikes = run_mnist_inference(path_to_backend, trained_network, dataset, log_path);
    std::cout << get_time_string() << ": inference finished  -- output spike count is " << spikes.size() << std::endl;

    // Evaluate results.
    inference_evaluation::InferenceResultsProcessor inference_processor;
    inference_processor.process_inference_results(spikes, dataset);

    inference_processor.write_inference_results_to_stream_as_csv(std::cout);

    return EXIT_SUCCESS;
}
