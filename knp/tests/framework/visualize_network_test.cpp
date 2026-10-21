/**
 * @file visualize_network_test.cpp
 * @brief Tests for network graph building and visualization.
 * @kaspersky_support Kirill L.
 * @date 05.08.2026
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

#include <knp/framework/backend_loader.h>
#include <knp/framework/model.h>
#include <knp/framework/network.h>
#include <knp/framework/population/creators.h>
#include <knp/framework/tags/name.h>
#include <knp/framework/visualizer/graph_builder.h>
#include <knp/framework/visualizer/graph_visualization.h>

#include <tests_common.h>

#include <cstddef>
#include <filesystem>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <variant>


// Helper function to create a simple network for testing.
knp::framework::Network create_test_network()
{
    knp::framework::Network network;

    // Create two populations.
    auto pop1 = knp::framework::population::creators::make_random<knp::neuron_traits::BLIFATNeuron>(5);
    auto pop2 = knp::framework::population::creators::make_random<knp::neuron_traits::BLIFATNeuron>(3);
    knp::framework::tags::set_name(pop1, "first");
    knp::framework::tags::set_name(pop2, "second");

    network.add_population(pop1);
    network.add_population(pop2);

    // Connect them.
    network.connect_populations<
        knp::synapse_traits::DeltaSynapse, knp::neuron_traits::BLIFATNeuron, knp::neuron_traits::BLIFATNeuron>(
        pop1, pop2);

    return network;
}


// Helper function to create a simple model for testing.
knp::framework::Model create_test_model()
{
    knp::framework::Network network = create_test_network();
    return knp::framework::Model(std::move(network));
}


// Helper function to get the DOT text of a graph.
std::string get_dot_text(const knp::framework::visualizer::NetworkGraph& graph)
{
    std::ostringstream out;
    knp::framework::visualizer::write_dot(graph, out);
    return out.str();
}


// Helper function to get the description of a graph.
std::string get_description(const knp::framework::visualizer::NetworkGraph& graph)
{
    std::ostringstream out;
    knp::framework::visualizer::print_network_description(graph, out);
    return out.str();
}


// Helper function to count lines of a text that match a pattern.
size_t count_lines(const std::string& text, const std::regex& pattern)
{
    std::istringstream in(text);
    size_t count = 0;
    for (std::string line; std::getline(in, line);)
    {
        if (std::regex_match(line, pattern)) ++count;
    }
    return count;
}


// Patterns of a node line and an edge line of a DOT text.
const std::regex dot_node_line(R"(\s*\d+ \[.*)");
const std::regex dot_edge_line(R"(\s*\d+ -> \d+ \[.*)");


TEST(VisualizeNetworkSuite, NetworkGraphConstruction)
{
    knp::framework::Network network = create_test_network();
    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(network);
    const std::string dot_text = get_dot_text(graph);

    // Check that we have correct number of nodes and edges.
    ASSERT_EQ(count_lines(dot_text, dot_node_line), 2u);
    ASSERT_EQ(count_lines(dot_text, dot_edge_line), 1u);
    ASSERT_FALSE(graph.is_dynamic());

    // Check node properties.
    ASSERT_NE(dot_text.find("0 [label=\"Population:\\nfirst\\nsize: 5\""), std::string::npos);
    ASSERT_NE(dot_text.find("1 [label=\"Population:\\nsecond\\nsize: 3\""), std::string::npos);

    // Check edge properties: the projection goes from the first population to the second one.
    ASSERT_NE(dot_text.find("0 -> 1 [label=\"Projection:\\n"), std::string::npos);
    ASSERT_NE(dot_text.find("\\nsize: 15\""), std::string::npos);  // 5 * 3 = 15 synapses.
}


TEST(VisualizeNetworkSuite, ModelGraphConstruction)
{
    knp::framework::Model model = create_test_model();
    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(model);
    const std::string dot_text = get_dot_text(graph);

    ASSERT_EQ(count_lines(dot_text, dot_node_line), 2u);
    ASSERT_EQ(count_lines(dot_text, dot_edge_line), 1u);
}


TEST(VisualizeNetworkSuite, DynamicGraphConstruction)
{
    knp::framework::Network network = create_test_network();

    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(knp::testing::get_backend_path());

    // Subscriptions are created when a backend starts, so the network is loaded and the backend runs zero steps.
    backend->load_all_populations(network.get_populations());
    backend->load_all_projections(network.get_projections());
    backend->start([](knp::core::Step) { return false; });

    knp::framework::visualizer::NetworkGraph graph =
        knp::framework::visualizer::build_graph(network, backend->get_message_endpoint());
    const std::string dot_text = get_dot_text(graph);

    // Populations come from the network, the projection comes from the bus subscriptions.
    ASSERT_TRUE(graph.is_dynamic());
    ASSERT_EQ(count_lines(dot_text, dot_node_line), 2u);
    ASSERT_NE(dot_text.find("0 -> 1 [label=\"Projection:\\n"), std::string::npos);
}


TEST(VisualizeNetworkSuite, ExternalEntitiesAreDrawn)
{
    knp::framework::Network network = create_test_network();

    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(knp::testing::get_backend_path());

    // A handler that listens to the first population. Such a subscription is never visible on the bus.
    const knp::core::UID handler_uid;
    const knp::core::UID population_uid =
        std::visit([](const auto& pop) { return pop.get_uid(); }, network.get_populations()[0]);

    knp::framework::visualizer::DynamicInfo info;
    info.bus_ = knp::framework::visualizer::get_bus_connections(backend->get_message_endpoint());
    info.entities_.push_back(knp::framework::visualizer::ExternalEntity{handler_uid, "WTA", {population_uid}});

    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(network, info);
    const std::string description = get_description(graph);

    // The handler is drawn with its own kind and receives spikes from the first population only.
    const size_t handler_position = description.find("    WTA #");
    ASSERT_NE(handler_position, std::string::npos);
    const size_t handler_line_end = description.find('\n', handler_position);
    const std::string handler_line = description.substr(handler_position, handler_line_end - handler_position);
    ASSERT_NE(handler_line.find(": receive from #Population 'first' (Spike Message); send to"), std::string::npos);
}


TEST(VisualizeNetworkSuite, WriteDotToStream)
{
    knp::framework::Network network = create_test_network();
    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(network);

    std::ostringstream out;
    knp::framework::visualizer::write_dot(graph, out);
    const std::string dot_text = out.str();

    ASSERT_NE(dot_text.find("digraph G {"), std::string::npos);
    ASSERT_NE(dot_text.find("0 -> 1"), std::string::npos);
    ASSERT_NE(dot_text.find("size: 15"), std::string::npos);

    // A static graph contains static information only: no dynamic color, no dotted edges, no invisible nodes.
    ASSERT_EQ(dot_text.find("yellow"), std::string::npos);
    ASSERT_EQ(dot_text.find("dotted"), std::string::npos);
    ASSERT_EQ(dot_text.find("invis"), std::string::npos);
}


TEST(VisualizeNetworkSuite, RenderGraphToStream)
{
    knp::framework::Network network = create_test_network();
    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(network);

    std::ostringstream out;
    ASSERT_TRUE(knp::framework::visualizer::render_graph(graph, out, "svg"));
    ASSERT_FALSE(out.str().empty());
}


TEST(VisualizeNetworkSuite, VisualizationFilesGeneration)
{
    knp::framework::Network network = create_test_network();
    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(network);

    const std::filesystem::path test_dir = std::filesystem::temp_directory_path() / "knp_test_visualization";
    std::filesystem::remove_all(test_dir);

    const auto dot_file = test_dir / "dot_files" / "graph_static.dot";
    knp::framework::visualizer::write_dot(graph, dot_file);
    ASSERT_TRUE(std::filesystem::exists(dot_file));
    ASSERT_GT(std::filesystem::file_size(dot_file), 0u);

    const auto image_file = test_dir / "png_files" / "graph_static.png";
    knp::framework::visualizer::render_graph(graph, image_file);
    ASSERT_TRUE(std::filesystem::exists(image_file));
    ASSERT_GT(std::filesystem::file_size(image_file), 0u);

    std::filesystem::remove_all(test_dir);
}


TEST(VisualizeNetworkSuite, NetworkGraphPrintFunctions)
{
    knp::framework::Network network = create_test_network();
    knp::framework::visualizer::NetworkGraph graph = knp::framework::visualizer::build_graph(network);

    // Test that print functions don't crash.
    EXPECT_NO_THROW(knp::framework::visualizer::print_network_description(graph))
        << "Print functions failed with exception";

    // The second population receives spikes from the first one.
    ASSERT_NE(
        get_description(graph).find("Population #1 'second' of size 3: receive from #Population 'first' (Projection: "),
        std::string::npos);
}
