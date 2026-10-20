/**
 * @file main.cpp
 * @brief Example of training a MNIST network.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
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
#include <knp/framework/visualizer/graph_visualization.h>

#include <iostream>

#include "knp/framework/transforms/upcasting.h"

#include "dataset.h"
#include "evaluate_results.h"
#include "inference.h"
#include "network_validation.h"
#include "parse_arguments.h"
#include "save_network.h"
#include "training.h"
#include "visualization.h"


/**
 * @brief Run only inference for model.
 *
 * @tparam Neuron neuron type.
 *
 * @param model_desc model description containing the path to the saved model.
 * @param dataset dataset used for inference.
 *
 * @return vector of spike messages produced during inference.
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> run_model_inference_only(const ModelDescription& model_desc,
                                                                         const Dataset& dataset)
{
    AnnotatedNetwork network;
    knp::framework::BackendLoader backend_loader;
    if (model_desc.model_saving_path_.empty())
        throw std::runtime_error("Model loading path is not defined for inference-only mode");

    // Perform inference using the pre-trained model.
    return infer_model<Neuron>(model_desc, dataset, network, backend_loader);
}


/**
 * @brief Run full training pipeline followed by inference.
 *
 * @details This function constructs the network, trains it, optionally upcasts populations and projections 
 * to their base types, and finally runs inference on the test set.
 *
 * @tparam Neuron neuron type.
 *
 * @param model_desc model description containing configuration parameters and paths.
 * @param dataset dataset used for training and inference.
 *
 * @return vector of output spike messages produced during the final inference step.
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> run_model_training(const ModelDescription& model_desc,
                                                                   const Dataset& dataset)
{
    AnnotatedNetwork network;
    knp::framework::BackendLoader backend_loader;

    // Build the network according to the selected neuron model.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    network = construct_network<Neuron>(model_desc);

    // Execute the training phase.
    auto bus_connections = train_model<Neuron>(model_desc, dataset, network, backend_loader);

    // Build the graphs.
    const auto static_graph = build_graph(network);
    const auto dynamic_graph = build_graph(network, bus_connections);

    // Print their descriptions to the log.
    knp::framework::visualizer::print_network_description(static_graph);
    knp::framework::visualizer::print_network_description(dynamic_graph);

    // Save the DOT and the image files.
    knp::framework::visualizer::save_dot(static_graph, "visualization_docs/dot_files/graph_static.dot");
    knp::framework::visualizer::save_image(static_graph, "visualization_docs/png_files/graph_static.png");
    
    knp::framework::visualizer::save_dot(dynamic_graph, "visualization_docs/dot_files/graph_dynamic.dot");
    knp::framework::visualizer::save_image(dynamic_graph, "visualization_docs/png_files/graph_dynamic.png");

    // Upcast populations to the most generic neuron type for serialization.
    if (model_desc.type_ == SupportedModelType::BLIFAT)
        knp::framework::upcast_populations<knp::neuron_traits::BLIFATNeuron>(network.network_);
    else if (model_desc.type_ == SupportedModelType::AltAI)
        knp::framework::upcast_populations<knp::neuron_traits::AltAILIF>(network.network_);
    else throw std::runtime_error("Unsupported model type");

    // Upcast projections to the generic `DeltaSynapse` type.
    knp::framework::upcast_projections<knp::synapse_traits::DeltaSynapse>(network.network_);

    // Perform inference on the trained model.
    return infer_model<Neuron>(model_desc, dataset, network, backend_loader);
}


/**
 * @brief Execute complete model pipeline for specified neuron type.
 *
 * @details This template function orchestrates the entire machine learning pipeline for neural networks, including
 * dataset processing, network construction, training, inference, and evaluation. It serves as the core execution
 * engine for both AltAI and BLIFAT neuron models.
 *
 * @tparam Neuron neuron type.
 *
 * @param model_desc model description containing configuration parameters and paths.
 */
template <typename Neuron>
void run_model(const ModelDescription& model_desc)
{
    Dataset dataset = process_dataset(model_desc);
    AnnotatedNetwork network;
    knp::framework::BackendLoader backend_loader;
    std::vector<knp::core::messaging::SpikeMessage> results;
    if (model_desc.inference_only_) results = run_model_inference_only<Neuron>(model_desc, dataset);
    else results = run_model_training<Neuron>(model_desc, dataset);

    // Evaluate and report inference results.
    evaluate_results(results, dataset);
}


/**
 * @brief Main application entry point.
 *
 * @details This function serves as the primary execution point for the MNIST neural network learning application.
 * It handles command-line argument parsing, configuration validation, user interaction, and routes execution to
 * the appropriate neuron model.
 *
 * @param argc argument count.
 * @param argv arguments values.
 * 
 * @return exit code.
 */
int main(int argc, char** argv)
{
    // Parse command-line arguments and validate configuration.
    std::optional<ModelDescription> model_desc_opt = parse_arguments(argc, argv);
    if (!model_desc_opt.has_value()) return EXIT_FAILURE;
    const ModelDescription& model_desc = model_desc_opt.value();

    // Display configuration to user for confirmation.
    std::cout << "Model description:\n"
              << model_desc << "\nPress ENTER to accept parameters and start model." << std::endl;
    std::cin.get();
    std::cout << "Starting model." << std::endl;

    // Execute model according to the selected neuron type.
    switch (model_desc.type_)
    {
        case SupportedModelType::BLIFAT:
        {
            // cppcheck-suppress throwInEntryPoint
            run_model<knp::neuron_traits::BLIFATNeuron>(model_desc);
            break;
        }
        case SupportedModelType::AltAI:
        {
            run_model<knp::neuron_traits::AltAILIF>(model_desc);
            break;
        }
        default:
            throw std::runtime_error("Unknown model type.");
    }

    return EXIT_SUCCESS;
}
