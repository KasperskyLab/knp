/**
 * @file model_desc.h
 * @brief Model description.
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

#pragma once

#include <filesystem>


/**
 * @brief Supported model types
 */
enum class SupportedModelType
{
    BLIFAT,
    AltAI
};


/*
 * @brief All parameters that may be changed from command line.
 */
struct ModelDescription
{
    // cppcheck-suppress unusedStructMember
    SupportedModelType type_;
    // cppcheck-suppress unusedStructMember
    size_t train_images_amount_, inference_images_amount_;
    std::filesystem::path images_file_path_, labels_file_path_, backend_path_, log_path_, model_saving_path_;
};


/*
 * @brief Printing helper for model description.
 * @param stream Stream.
 * @param desc Model desciption.
 * @return Stream.
 */
std::ostream& operator<<(std::ostream& stream, ModelDescription const& desc);
