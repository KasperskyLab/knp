#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/framework/model.h>
#include <knp/framework/network.h>
#include <knp/framework/population/creators.h>
#include <knp/framework/projection/connectors.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <tests_common.h>

#include <filesystem>
#include <fstream>
#include <string>

using BLIFATParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using Synapse = DeltaProjection::Synapse;

// Helper function to create a simple network for testing
knp::framework::Network create_test_network()
{
    knp::framework::Network network;

    // Create two populations
    auto pop1 = knp::framework::population::creators::make_random<knp::neuron_traits::BLIFATNeuron>(5);
    auto pop2 = knp::framework::population::creators::make_random<knp::neuron_traits::BLIFATNeuron>(3);

    network.add_population(pop1);
    network.add_population(pop2);

    // Connect them
    auto proj_uid = network.connect_populations<
        knp::synapse_traits::DeltaSynapse, knp::neuron_traits::BLIFATNeuron, knp::neuron_traits::BLIFATNeuron>(
        pop1, pop2);

    return network;
}

// Helper function to create a simple model for testing
knp::framework::Model create_test_model()
{
    knp::framework::Network network = create_test_network();
    return knp::framework::Model(std::move(network));
}

TEST(VisualizeNetworkSuite, NetworkGraphConstruction)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph(network);

    // Check that we have correct number of nodes and edges
    ASSERT_EQ(graph.nodes_.size(), 2);
    ASSERT_EQ(graph.edges_.size(), 1);

    // Check node properties
    ASSERT_EQ(graph.nodes_[0].size_, 5);
    ASSERT_EQ(graph.nodes_[1].size_, 3);

    // Check edge properties
    ASSERT_EQ(graph.edges_[0].size_, 15);  // 5 * 3 = 15 synapses
    ASSERT_EQ(graph.edges_[0].index_from_, 0);
    ASSERT_EQ(graph.edges_[0].index_to_, 1);
}

TEST(VisualizeNetworkSuite, NetworkGraphNodeAndEdgeAccess)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph(network);

    // Test accessing nodes
    ASSERT_FALSE(graph.nodes_.empty());
    ASSERT_EQ(graph.nodes_.size(), 2);

    // Test accessing edges
    ASSERT_FALSE(graph.edges_.empty());
    ASSERT_EQ(graph.edges_.size(), 1);

    // Test node properties
    const auto& node0 = graph.nodes_[0];
    const auto& node1 = graph.nodes_[1];

    ASSERT_GT(node0.size_, 0);
    ASSERT_GT(node1.size_, 0);
    ASSERT_FALSE(node0.name_.empty());
    ASSERT_FALSE(node1.name_.empty());

    // Test edge properties
    const auto& edge = graph.edges_[0];
    ASSERT_GT(edge.size_, 0);
    ASSERT_FALSE(edge.name_.empty());
    ASSERT_EQ(edge.index_from_, 0);
    ASSERT_EQ(edge.index_to_, 1);
}


TEST(VisualizeNetworkSuite, StaticVisualizationFilesGeneration)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph(network);

    // Set up paths for testing
    std::string test_dir = "./test_visualization/";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directories(test_dir);

    // Change the default path for testing
    knp::framework::set_saving_path(test_dir);

    // Generate dot file
    knp::framework::ConfigVisualizePathes file_info{};
    std::string dot_file = file_info.name_dot_file;
    std::string png_file = file_info.name_png_file;

    // Test that we can create dot file
    try
    {
        // Verify that the graph has expected structure
        ASSERT_EQ(graph.nodes_.size(), 2);
        ASSERT_EQ(graph.edges_.size(), 1);

        // Verify that the paths are correctly formed
        ASSERT_FALSE(dot_file.empty());
        ASSERT_FALSE(png_file.empty());

        // Verify that directories exist
        std::filesystem::path dot_dir = std::filesystem::path(dot_file).parent_path();
        std::filesystem::path png_dir = std::filesystem::path(png_file).parent_path();

        ASSERT_TRUE(std::filesystem::exists(dot_dir));
        ASSERT_TRUE(std::filesystem::exists(png_dir));
        ASSERT_TRUE(std::filesystem::is_directory(dot_dir));
        ASSERT_TRUE(std::filesystem::is_directory(png_dir));

        // Verify that the files don't exist yet (they will be created during visualization)
        ASSERT_FALSE(std::filesystem::exists(dot_file));
        ASSERT_FALSE(std::filesystem::exists(png_file));

        // Test actual visualization function to ensure files are created
        knp::framework::visualize_network(network);

        // Verify that files were created
        ASSERT_TRUE(std::filesystem::exists(dot_file));
        ASSERT_TRUE(std::filesystem::exists(png_file));

        // Verify that files are not empty
        ASSERT_GT(std::filesystem::file_size(dot_file), 0);
        ASSERT_GT(std::filesystem::file_size(png_file), 0);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Failed to generate visualization files: " << e.what();
    }
}


TEST(VisualizeNetworkSuite, DynamicVisualizationWithBackend)
{
    knp::framework::Network network = create_test_network();
    knp::framework::Model model(std::move(network));

    // Create backend using the correct API from backend_loader_test.cpp
    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(knp::testing::get_backend_path());

    // Test that we can call visualization functions without crashing
    try
    {
        // This should not crash
        knp::framework::visualize_network(model.get_network(), backend);
        knp::framework::visualize_bus(model.get_network(), backend);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Dynamic visualization failed with exception: " << e.what();
    }
}

TEST(VisualizeNetworkSuite, ModelVisualization)
{
    knp::framework::Model model = create_test_model();

    // Test that we can visualize model directly
    try
    {
        knp::framework::visualize_network(model);
        // knp::framework::visualize_network(model, nullptr); // Test with null backend
    }
    catch (const std::exception& e)
    {
        FAIL() << "Model visualization failed with exception: " << e.what();
    }
}


TEST(VisualizeNetworkSuite, NetworkGraphPrintFunctions)
{
    knp::framework::Network network = create_test_network();
    knp::framework::NetworkGraph graph(network);

    // Test that print functions don't crash
    try
    {
        // These functions write to stdout, so we just ensure they don't throw
        knp::framework::print_network_description(graph);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Print functions failed with exception: " << e.what();
    }
}

TEST(VisualizeNetworkSuite, VisualizationPathConfiguration)
{
    // Test setting custom visualization paths
    std::string custom_path = "/tmp/custom_visualization/";
    std::filesystem::create_directories(custom_path);

    knp::framework::set_saving_path(custom_path);

    // Verify default path was changed
    knp::framework::ConfigVisualizePathes config;
    ASSERT_EQ(config.dir, custom_path);
}
