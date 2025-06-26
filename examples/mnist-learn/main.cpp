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

#include <filesystem>
#include <functional>
#include <iostream>

#include "data_read.h"
#include "evaluation.h"
#include "inference.h"
#include "time_string.h"
#include "train.h"


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Not enough parameters.\n First parameter: path to frames file.\n "
                     "Second parameter: path to labels file.\n Third parameter (optional) path to log output directory."
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::filesystem::path log_path;
    if (argc >= 4) log_path = argv[3];

    // Defines path to backend, on which to run a network.
    std::filesystem::path path_to_backend =
        std::filesystem::path(argv[0]).parent_path() / "knp-cpu-multi-threaded-backend";

    // Read data from corresponding files.
    auto spike_frames = read_spike_frames(argv[1]);
    auto labels = read_labels(argv[2], learning_period);

    // Construct network and run training.
    AnnotatedNetwork trained_network = train_mnist_network(path_to_backend, spike_frames, labels.train_, log_path);

    // Run inference for the same network.
    auto spikes = run_mnist_inference(path_to_backend, trained_network, spike_frames, log_path);
    std::cout << get_time_string() << ": inference finished  -- output spike count is " << spikes.size() << std::endl;

    // Evaluate results.
    process_inference_results(spikes, labels.test_, testing_period);
    return EXIT_SUCCESS;
}
