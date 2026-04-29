/**
 * @file data_storage_hdf5.h
 * @brief Load data from file.
 * @kaspersky_support Vartenkov Andrey
 * @date 22.04.2024
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
 * @brief Storage namespace.
 */
namespace knp::framework::io::storage
{

/**
 * @brief Data storage namespace.
 */
namespace native
{

// TODO: Return and get iterators.

/**
 * @brief Read spike messages from an HDF5 file.
 * 
 * @param path_to_h5 path to HDF5 data file.
 * @param uid sender UID.
 * @param time_per_step time per step.
 * @param strict_format if `true`, method throws exception on wrong format.
 * 
 * @return vector of messages sorted by timestamps.
 * 
 * @details For example, if @p time_per_step is set to `1`, then spike messages with timestamps
 * `0` and `0.5` will be a part of the same message.
 */
KNP_DECLSPEC std::vector<core::messaging::SpikeMessage> load_messages_from_h5(
    const std::filesystem::path &path_to_h5, const knp::core::UID &uid, float time_per_step = 1.0f,
    bool strict_format = true);


/**
 * @brief Save a vector of spike messages to HDF5 file.
 * 
 * @param messages vector of spike messages to save.
 * @param path_to_save path to file.
 * @param time_per_step time per step.
 * 
 * @details If you use steps as a time unit by default, we recommend setting @p time_per_step to `1`.
 * 
 * @note Passing messages by value is not an error. Messages are sorted inside the function.
 */
KNP_DECLSPEC void save_messages_to_h5(
    const std::vector<core::messaging::SpikeMessage> &messages, const std::filesystem::path &path_to_save,
    float time_per_step = 1.0f);

}  // namespace native

}  // namespace knp::framework::io::storage
