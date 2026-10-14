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
#include <knp/framework/visualizer/visualize_network.h>

#include <tests_common.h>

#include <filesystem>
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


TEST(VisualizeNetworkSuite, NetworkGraphConstruction)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph = knp::framework::build_graph(network);

    // Check that we have correct number of nodes and edges.
    ASSERT_EQ(boost::num_vertices(graph.graph_), 2u);
    ASSERT_EQ(boost::num_edges(graph.graph_), 1u);
    ASSERT_FALSE(graph.is_dynamic_);

    // Check node properties.
    auto vertex_from = graph.find_node(graph.graph_[0].uid_);
    ASSERT_TRUE(vertex_from.has_value());
    ASSERT_EQ(graph.graph_[0].size_.value(), 5u);
    ASSERT_EQ(graph.graph_[1].size_.value(), 3u);
    ASSERT_FALSE(graph.graph_[0].name_.empty());
    ASSERT_EQ(graph.graph_[0].node_type_, "Population");

    // Check edge properties.
    auto [edge_it, edge_end] = boost::edges(graph.graph_);
    const auto& edge = graph.graph_[*edge_it];
    ASSERT_EQ(edge.size_.value(), 15u);  // 5 * 3 = 15 synapses.
    ASSERT_TRUE(edge.is_static_);
    ASSERT_TRUE(edge.index_from_.has_value());
    ASSERT_TRUE(edge.index_to_.has_value());
    ASSERT_TRUE(edge.index_from_.value() == graph.graph_[0].uid_);
    ASSERT_TRUE(edge.index_to_.value() == graph.graph_[1].uid_);
    ASSERT_EQ(boost::source(*edge_it, graph.graph_), 0u);
    ASSERT_EQ(boost::target(*edge_it, graph.graph_), 1u);

    // Check that the projection is indexed by UID.
    ASSERT_EQ(graph.find_edges(edge.uid_).size(), 1u);
}


TEST(VisualizeNetworkSuite, ModelGraphConstruction)
{
    knp::framework::Model model = create_test_model();
    knp::framework::NetworkGraph graph = knp::framework::build_graph(model);

    ASSERT_EQ(boost::num_vertices(graph.graph_), 2u);
    ASSERT_EQ(boost::num_edges(graph.graph_), 1u);
}


TEST(VisualizeNetworkSuite, DynamicGraphConstruction)
{
    knp::framework::Network network = create_test_network();

    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(knp::testing::get_backend_path());

    knp::framework::NetworkGraph graph = knp::framework::build_graph(network, backend->get_message_endpoint());

    // The network structure is always present, the bus can only extend it.
    ASSERT_TRUE(graph.is_dynamic_);
    ASSERT_GE(boost::num_vertices(graph.graph_), 2u);
    ASSERT_GE(boost::num_edges(graph.graph_), 1u);
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

    knp::framework::DynamicInfo info;
    info.bus_ = knp::framework::get_bus_connections(backend->get_message_endpoint());
    info.entities_.push_back(knp::framework::ExternalEntity{handler_uid, "WTA", {population_uid}});

    knp::framework::NetworkGraph graph = knp::framework::build_graph(network, info);

    auto handler_vertex = graph.find_node(handler_uid);
    ASSERT_TRUE(handler_vertex.has_value());
    ASSERT_EQ(graph.graph_[handler_vertex.value()].node_type_, "WTA");
    ASSERT_EQ(boost::in_degree(handler_vertex.value(), graph.graph_), 1u);
}


TEST(VisualizeNetworkSuite, WriteDotToStream)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph = knp::framework::build_graph(network);

    std::ostringstream out;
    knp::framework::write_dot(graph, out);
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
    knp::framework::NetworkGraph graph = knp::framework::build_graph(network);

    std::ostringstream out;
    ASSERT_TRUE(knp::framework::render_graph(graph, out, "svg"));
    ASSERT_FALSE(out.str().empty());
}


TEST(VisualizeNetworkSuite, VisualizationFilesGeneration)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph = knp::framework::build_graph(network);

    const std::filesystem::path test_dir = std::filesystem::temp_directory_path() / "knp_test_visualization";
    std::filesystem::remove_all(test_dir);

    const auto image_file = knp::framework::visualize(graph, "graph_static", test_dir);

    const auto dot_file = test_dir / "dot_files" / "graph_static.dot";
    ASSERT_TRUE(std::filesystem::exists(dot_file));
    ASSERT_GT(std::filesystem::file_size(dot_file), 0u);

    ASSERT_FALSE(image_file.empty());
    ASSERT_TRUE(std::filesystem::exists(image_file));
    ASSERT_GT(std::filesystem::file_size(image_file), 0u);

    std::filesystem::remove_all(test_dir);
}


TEST(VisualizeNetworkSuite, NetworkGraphPrintFunctions)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph = knp::framework::build_graph(network);

    // Test that print functions don't crash.
    EXPECT_NO_THROW(knp::framework::print_network_description(graph)) << "Print functions failed with exception";
}
