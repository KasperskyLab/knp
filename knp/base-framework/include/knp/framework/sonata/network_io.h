/**
 * @file network_io.h
 * @brief Saving and loading networks.
 * @kaspersky_support A. Vartenkov
 * @date 31.01.2024
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

#pragma once
#include <knp/core/impexp.h>
#include <knp/framework/network.h>

#include <filesystem>

/**
 * @brief SONATA namespace.
 */
namespace knp::framework::sonata
{
/**
 * @brief Save network to disk.
 * 
 * @param network network to save.
 * @param dir directory to save the network.
 * 
 * @note The network is saved in the SONATA format.
 */
KNP_DECLSPEC void save_network(const Network &network, const std::filesystem::path &dir);


/**
 * @brief Load network from disk.
 * 
 * @param config_path path to network configuration file.
 * 
 * @return loaded network.
 */
KNP_DECLSPEC Network load_network(const std::filesystem::path &config_path);

}  // namespace knp::framework::sonata
