/**
 * @file data_storage_json.h
 * @brief Load data from file.
 * @kaspersky_support Vartenkov Andrey
 * @date 16.04.2024
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
#include <knp/core/messaging/messaging.h>

#include <filesystem>
#include <vector>


/**
 * @brief Data storage namespace.
 */
namespace knp::framework::io::storage::native
{

// TODO: return and get iterators.

/**
 * @brief Read spike messages from a JSON file.
 * 
 * @param path_to_json path to JSON data file.
 * @param uid sender UID.
 * @param strict_format if `true`, method throws exception on wrong format.
 * 
 * @return vector of messages sorted by timestamps.
 */
KNP_DECLSPEC std::vector<core::messaging::SpikeMessage> load_messages_from_json(
    const std::filesystem::path &path_to_json, const knp::core::UID &uid, bool strict_format = true);


/**
 * @brief Save a vector of spike messages to JSON file.
 * 
 * @param messages vector of spike messages to save.
 * @param path_to_save path to file.
 * 
 * @note Passing messages by value is not an error. Messages are sorted inside the function.
 */
KNP_DECLSPEC void save_messages_to_json(
    const std::vector<core::messaging::SpikeMessage> &messages, const std::filesystem::path &path_to_save);
}  // namespace knp::framework::io::storage::native
