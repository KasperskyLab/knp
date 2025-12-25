/**
 * @file main.cpp
 * @brief A program for building and running MNIST-classification spiking network.
 * @kaspersky_support A. Vartenkov
 * @date 26.07.2024
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

#include <knp/framework/io/input_channel.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>
#include <knp/framework/sonata/network_io.h>

#include <iostream>

#include <boost/program_options.hpp>

#include "inference.h"
#include "visualize_network.h"


// Namespace for program options.
namespace po = boost::program_options;


// Load network and show or run it depending on command line parameters.
// On `task=show`, the function loads a network and shows a subgraph of it.
// On `task=infer`, the function loads and runs network using data from a binary data file.
int main(int argc, char **argv)
{
    // Creates an output ID.
    knp::core::UID output_uid;
    // Defines `std::string` object for task types.
    std::string task;
    // Defines `std::string` objects for path to network and path to a file storing data.
    std::string path_to_network, path_to_data;
    po::options_description options;
    // Defines options for task types, path to network, and path to a file storing data.
    options.add_options()("help,h", "Produce help message.")(
        "task,t", po::value(&task), "Type of task: show, train, infer.")(
        "net-path,p", po::value(&path_to_network), "File or directory for network storage.")(
        "data-path,d", po::value(&path_to_data), "File for data storage.");
    // Stores defines options in a variable map.
    po::variables_map options_map;
    po::store(po::parse_command_line(argc, argv, options), options_map);
    po::notify(options_map);

    // If a path to network is not provided, the function exits.
    if (!options_map.count("net-path"))
    {
        std::cerr << "No \"net-path\" parameter required to load network. Stopping the program..." << std::endl;
        return EXIT_FAILURE;
    }

    // If `task=show`, the function loads a network and shows a subgraph of it.
    if (task == "show")
    {
        const auto network_path = options_map["net-path"].as<std::string>();
        // Creates a network objects by loading it using the `load_network` function from the `sonata` namespace.
        // Online Help links:
        // https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
        // https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=274991
        knp::framework::Network network = knp::framework::sonata::load_network(network_path);
        // Constructs a network graph.
        const NetworkGraph net_graph(network);
        // Prints descriptions of graph connections.
        print_network_description(net_graph);
        // Draws a subgraph in the OpenCV window.
        // Press `Esc` to exit the OpenCV window.
        position_network_test(NetworkGraph(network), divide_graph_by_connectivity(net_graph)[0], {1000, 700});
    }

    // If `task=infer`, the function loads a network and runs inference.
    if (task == "infer")
    {
        // Defines path to backend, on which to run a network.
        std::filesystem::path path_to_backend =
            std::filesystem::path(argv[0]).parent_path() / "knp-cpu-single-threaded-backend";
        // Runs inference of network on the specified path.
        do_inference(
            options_map["net-path"].as<std::string>(), options_map["data-path"].as<std::string>(), path_to_backend);
    }

    return EXIT_SUCCESS;
}
