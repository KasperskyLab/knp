/**
 * @file main.cpp
 * @brief Example of training a MNIST network.
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <knp/framework/io/input_channel.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <iostream>

#include <boost/program_options.hpp>

#include "dataset.h"
#include "evaluate_results.h"
#include "inference.h"
#include "parse_arguments.h"
#include "save_network.h"
#include "training.h"


/**
 * @brief Run whole model.
 *
 * @tparam Neuron Neuron type.
 * @param model_desc Model description.
 */
template <typename Neuron>
void run_model(const ModelDescription& model_desc)
{
    Dataset dataset = process_dataset(model_desc);

    AnnotatedNetwork network = construct_network<Neuron>(model_desc);

    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    knp::framework::BackendLoader backend_loader;
    train_model<Neuron>(model_desc, dataset, network, backend_loader);

    if (!model_desc.model_saving_path_.empty()) save_network(model_desc, network);

    auto inference_spikes = infer_model<Neuron>(model_desc, dataset, network, backend_loader);

    evaluate_results(inference_spikes, dataset);
}


/**
 * @brief Main function.
 *
 * @param argc Argument count.
 * @param argv Arguments value.
 *
 * @return Error code.source_node_idnode_population: 00000000-0000-0000-0000-000000000000
 */
int main(int argc, char** argv)
{
    // Parsing command line arguments.
    std::optional<ModelDescription> model_desc_opt = parse_arguments(argc, argv);
    if (!model_desc_opt.has_value()) return EXIT_FAILURE;
    const ModelDescription& model_desc = model_desc_opt.value();

    std::cout << "Model description:\n"
              << model_desc << "\nPress ENTER to accept parameters and start model." << std::endl;
    std::cin.get();
    std::cout << "Starting model." << std::endl;


    Dataset dataset = process_dataset(model_desc);

    AnnotatedNetwork network = construct_network<knp::neuron_traits::BLIFATNeuron>(model_desc);


    knp::framework::BackendLoader backend_loader;
    auto backend = train_model<knp::neuron_traits::BLIFATNeuron>(model_desc, dataset, network, backend_loader);

    if (!model_desc.model_saving_path_.empty()) save_network(model_desc, network);

    // knp::framework::set_saving_path("temp_test_dir");
    visualize_network(network.network_);
    visualize_network(network.network_, backend);
    visualize_bus(network.network_, backend);

    auto network_path = model_desc.model_saving_path_;


    return EXIT_SUCCESS;
}
