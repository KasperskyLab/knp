#pragma once
#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

extern "C"
{
#include <graphviz/gvc.h>
}


/**
 * @brief Create DOT file for static network
 *
 * @param file_name output file name
 * @param graph network graph
 */
void create_dot_file_for_static_network(std::string& file_name, knp::framework::NetworkGraph& graph);

/**
 * @brief Create DOT file for bus messages
 *
 * @param file_name output DOT file name
 * @param graph network graph
 * @param backend shared pointer to backend
 */
void create_dot_file_for_bus(
    const std::string& file_name, const knp::framework::NetworkGraph& graph,
    std::shared_ptr<knp::core::Backend>& backend);

/**
 * @brief Create DOT file for dynamic network
 *
 * @param file_name output file name
 * @param graph network graph
 * @param backend shared pointer to backend
 */
void create_dot_file_for_dynamic_network(
    const std::string& file_name, const knp::framework::NetworkGraph& graph,
    std::shared_ptr<knp::core::Backend>& backend);

/**
 * @brief Convert DOT file to PNG image
 *
 * @param path_to_dot_file path to input DOT file
 * @param path_to_png_file path to output PNG file
 *
 * @return boolean indicating success
 */
bool convert_dot_to_png(const std::string& path_to_dot_file, const std::string& path_to_png_file);

/**
 * @brief Create PNG file from DOT file
 *
 * @param dot_file path to DOT file
 * @param png_file path to PNG file
 */
void create_png_file(const std::string& dot_file, const std::string& png_file);
