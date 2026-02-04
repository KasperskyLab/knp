/**
 * @file main.cpp
 * @brief Example of training a MNIST network.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <iostream>

#include "dataset.h"
#include "model.h"
#include "parse_arguments.h"
#include "save_network.h"


int main(int argc, char** argv)
{
    std::optional<ModelDescription> model_desc_opt = parse_arguments(argc, argv);
    if (!model_desc_opt.has_value()) return EXIT_FAILURE;
    const ModelDescription& model_desc = model_desc_opt.value();

    std::cout << "Starting model:\n" << model_desc << std::endl;

    Dataset dataset = process_dataset(model_desc);
    std::cout << "Processed dataset, training will last " << dataset.get_steps_amount_for_training()
              << " steps, inference " << dataset.get_steps_amount_for_inference() << " steps\n"
              << std::endl;

    AnnotatedNetwork network = construct_network(model_desc);

    train_network(network, model_desc, dataset);

    save_network(model_desc, network);

    auto inference_spikes = run_inference_on_network(network, model_desc, dataset);

    // Evaluate results.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=306417
    knp::framework::inference_evaluation::classification::InferenceResultsProcessor inference_processor;
    inference_processor.process_inference_results(inference_spikes, dataset);

    inference_processor.write_inference_results_to_stream_as_csv(std::cout);
}
