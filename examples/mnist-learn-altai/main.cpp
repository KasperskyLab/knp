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
constexpr size_t images_amount_to_train = 1000;
constexpr float dataset_split = 0.8;
constexpr size_t classes_amount = 10;

namespace data_processing = knp::framework::data_processing::classification::images;
namespace inference_evaluation = knp::framework::inference_evaluation::classification;
/*
CLASS,TOTAL_VOTES,TRUE_POSITIVES,FALSE_NEGATIVES,FALSE_POSITIVES,TRUE
_NEGATIVES,PRECISION,RECALL,PREVALENCE,ACCURACY,F_SCORE

0,21,0,21,0,229,0,0,0.084,0.916,0
1,26,0,26,0,224,0,0,0.104,0.896,0
2,23,0,23,0,227,0,0,0.092,0.908,0
3,23,0,23,0,227,0,0,0.092,0.908,0
4,23,0,23,0,227,0,0,0.092,0.908,0
5,29,0,29,0,221,0,0,0.116,0.884,0
6,24,0,24,0,226,0,0,0.096,0.904,0
7,30,0,30,0,220,0,0,0.12,0.88,0
8,22,0,22,0,228,0,0,0.088,0.912,0
9,29,0,29,0,221,0,0,0.116,0.884,0

0,21,19,0,2,229,0.904762,0.904762,0.076,0.992,0.904762
1,26,24,1,1,224,0.96,0.96,0.1,0.992,0.96
2,23,19,0,4,227,0.826087,0.826087,0.076,0.984,0.826087
3,23,23,0,0,227,1,1,0.092,1,1
4,23,21,0,2,227,0.913043,0.913043,0.084,0.992,0.913043
5,29,17,1,11,221,0.607143,0.607143,0.072,0.952,0.607143
6,24,19,0,5,226,0.791667,0.791667,0.076,0.98,0.791667
7,30,27,0,3,220,0.9,0.9,0.108,0.988,0.9
8,22,14,0,8,228,0.636364,0.636364,0.056,0.968,0.636364
9,29,11,2,16,221,0.407407,0.407407,0.052,0.928,0.407407


*/
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
